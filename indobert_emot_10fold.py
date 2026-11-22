
import argparse
import os
import random
import re
import time

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

# 1. KONFIGURASI
BASE_URL = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/"
FILES = {"train": "train_preprocess.csv", "valid": "valid_preprocess.csv", "test": "test_preprocess.csv"}
LABELS = ["anger", "fear", "happy", "love", "sadness"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_ALIAS = "indobert"
OUT = "outputs/indobert"
DATA = "data"             # train_preprocess.csv, valid_preprocess.csv, test_preprocess.csv 


def parse_args():
    p = argparse.ArgumentParser(description="IndoBERT + 10-Fold CV pada dataset EmoT")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="2 epoch, untuk laptop tanpa GPU")
    p.add_argument("--resume", action="store_true", help="lewati fold yang sudah tersimpan di outputs/")
    p.add_argument("--skip-final", action="store_true", help="tidak melatih model akhir untuk data uji")
    args = p.parse_args()
    if args.quick:
        args.epochs = 2
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 2. PERSIAPAN DATASET
def read_emot_csv(name):
    local = os.path.join(DATA, FILES[name])
    if os.path.exists(local):
        df = pd.read_csv(local)
    else:
        df = pd.read_csv(BASE_URL + FILES[name])
        os.makedirs(DATA, exist_ok=True)
        df.to_csv(local, index=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    label_col = next(c for c in df.columns if set(df[c].astype(str).str.strip()) <= set(LABELS))
    text_col = next(c for c in df.columns if c != label_col)
    df = df.rename(columns={text_col: "tweet", label_col: "label"})[["tweet", "label"]]
    df["label"] = df["label"].astype(str).str.strip()
    return df


def clean_text(text):
    t = str(text).lower()
    t = re.sub(r"\[username\]|\[url\]|\[sensitive-no\]", " ", t)  # token samaran bawaan dataset
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)      # "senenggg" -> "seneng"
    t = re.sub(r"[^a-z0-9\s.,!?'-]", " ", t)    # emoji dan simbol
    t = re.sub(r"([.,!?])\1+", r"\1", t)         # "!!!" -> "!"
    return re.sub(r"\s+", " ", t).strip()


def load_data():
    train, valid, test = (read_emot_csv(n) for n in ["train", "valid", "test"])
    cv_df = pd.concat([train, valid], ignore_index=True)
    for df in (cv_df, test):
        df["text"] = df["tweet"].map(clean_text)
    n0 = len(cv_df)
    cv_df = cv_df[cv_df.text.str.len() > 0].drop_duplicates("text")
    cv_df = cv_df[~cv_df.text.isin(set(test.text))].reset_index(drop=True)  # cegah kebocoran ke data uji
    test = test[test.text.str.len() > 0].reset_index(drop=True)
    print(f"Data CV: {n0} -> {len(cv_df)} setelah hapus kosong/duplikat/kebocoran | Data uji: {len(test)}")
    return cv_df, test


def describe_data(cv_df, test_df, tokenizer, max_len):
    rows = []
    for name, df in [("CV (latih+validasi)", cv_df), ("Uji", test_df)]:
        vc = df.label.value_counts().reindex(LABELS).fillna(0).astype(int)
        rows.append({"bagian": name, **{l: f"{vc[l]} ({vc[l] / len(df):.1%})" for l in LABELS}, "total": len(df)})
    pd.DataFrame(rows).to_csv(f"{OUT}/distribusi_label.csv", index=False)

    contoh = cv_df.groupby("label").sample(2, random_state=0)
    contoh[["tweet", "text", "label"]].to_csv(f"{OUT}/contoh_data.csv", index=False)

    n_tok = pd.Series([len(tokenizer.encode(t)) for t in cv_df.text])
    stat = n_tok.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1)
    stat["persen_terpotong"] = round((n_tok > max_len).mean() * 100, 2)
    stat.to_frame("panjang_token").to_csv(f"{OUT}/statistik_token.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Panjang token: median {stat['50%']}, persentil-95 {stat['95%']}, "
          f"terpotong pada max_len={max_len}: {stat['persen_terpotong']}%")


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), max_length=max_len, truncation=True)
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


# 3. TRAINING DAN EVALUASI
def build_model(model_name, device):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(device)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits               # forward pass
        loss = loss_fn(logits, labels)                # cross-entropy
        loss.backward()                               # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                              # perbarui bobot
        scheduler.step()                              # perbarui learning rate
        optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels")
        probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu())
    probs = torch.cat(probs).numpy()
    return probs.argmax(1), probs


def compute_metrics(y_true, y_pred):
    m = {"akurasi": accuracy_score(y_true, y_pred),
         "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
         "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
         "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    per_class = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    m.update({f"f1_{l}": v for l, v in zip(LABELS, per_class)})
    return m


def fine_tune(train_df, eval_df, tokenizer, args, device):
    set_seed(args.seed)
    model = build_model(MODEL_NAME, device)
    collate = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(EmotionDataset(train_df.text, train_df.label, tokenizer, args.max_len),
                              batch_size=args.batch, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(EmotionDataset(eval_df.text, eval_df.label, tokenizer, args.max_len),
                             batch_size=args.batch * 2, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    y_eval = eval_df.label.map(LABEL2ID).values

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        y_pred, probs = predict(model, eval_loader, device)
        f1 = f1_score(y_eval, y_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": loss, "eval_f1_macro": f1, "detik": time.time() - t0})
        print(f"   epoch {epoch}/{args.epochs} | loss {loss:.4f} | macro F1 {f1:.4f} | {time.time() - t0:.0f} dtk")
    return model, y_pred, probs, history


def cross_validate(cv_df, tokenizer, args, device):
    path = f"{OUT}/hasil_per_fold.csv"
    done = pd.read_csv(path) if (args.resume and os.path.exists(path)) else pd.DataFrame()
    results = done.to_dict("records")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr_idx, te_idx) in enumerate(skf.split(cv_df.text, cv_df.label), start=1):
        if not done.empty and k in set(done.fold):
            print(f"Fold {k}: sudah ada, dilewati")
            continue
        tr, te = cv_df.iloc[tr_idx], cv_df.iloc[te_idx]
        print(f"\n=== Pengujian ke-{k}/{args.folds} | latih {len(tr)} | uji {len(te)} ===")
        t0 = time.time()
        model, y_pred, _, history = fine_tune(tr, te, tokenizer, args, device)
        res = {"fold": k, "n_latih": len(tr), "n_uji": len(te),
               **compute_metrics(te.label.map(LABEL2ID).values, y_pred),
               "menit": (time.time() - t0) / 60}
        results.append(res)
        pd.DataFrame(history).assign(fold=k).to_csv(f"{OUT}/riwayat_fold_{k}.csv", index=False)
        pd.DataFrame(results).sort_values("fold").to_csv(path, index=False)  # simpan setiap fold selesai
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(results).sort_values("fold").reset_index(drop=True)


def baseline_cv(cv_df, args):
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(skf.split(cv_df.text, cv_df.label), start=1):
        tr, te = cv_df.iloc[tr_idx], cv_df.iloc[te_idx]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(vec.fit_transform(tr.text), tr.label)
        pred = clf.predict(vec.transform(te.text))
        rows.append({"fold": k, **compute_metrics(te.label.map(LABEL2ID).values, pd.Series(pred).map(LABEL2ID).values)})
    return pd.DataFrame(rows)


def summarize(bert, base, args):
    metrics = ["akurasi", "precision_macro", "recall_macro", "f1_macro"] + [f"f1_{l}" for l in LABELS] 
    summary = pd.DataFrame({"metrik": metrics,
                            "BERT_rata2": [bert[m].mean() for m in metrics],
                            "BERT_std": [bert[m].std() for m in metrics],
                            "Baseline_rata2": [base[m].mean() for m in metrics],
                            "Baseline_std": [base[m].std() for m in metrics]}).round(4)
    summary.to_csv(f"{OUT}/ringkasan_cv.csv", index=False)

    d = bert.set_index("fold").f1_macro - base.set_index("fold").f1_macro
    d = d.dropna()
    k = len(d)
    ratio = 1 / (args.folds - 1)          # n_uji / n_latih pada k-fold
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + ratio) * var) if var > 0 else float("inf")
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    uji = pd.DataFrame([{"selisih_rata2_f1_macro": d.mean(), "t_terkoreksi": t, "p_value": p, "jumlah_fold": k}]).round(5)
    uji.to_csv(f"{OUT}/uji_signifikansi.csv", index=False)
    print("\n=== Ringkasan 10-Fold CV ===")
    print(summary.to_string(index=False))
    print(uji.to_string(index=False))

    plt.figure(figsize=(8, 4))
    x = np.arange(len(bert))
    plt.bar(x - 0.2, bert.f1_macro, 0.4, label=MODEL_ALIAS)
    plt.bar(x + 0.2, base.set_index("fold").loc[bert.fold, "f1_macro"], 0.4, label="TF-IDF + LogReg")
    plt.axhline(bert.f1_macro.mean(), ls="--", lw=1, color="tab:blue")
    plt.xticks(x, [f"ke-{f}" for f in bert.fold])
    plt.ylabel("macro F1"); plt.title("Macro F1 per pengujian (fold)"); plt.legend()
    plt.savefig(f"{OUT}/f1_per_fold.png", dpi=150, bbox_inches="tight"); plt.close()
    return summary


def final_test(cv_df, test_df, tokenizer, args, device):
    print("\n=== Model akhir: latih semua data CV, uji pada data uji terpisah ===")
    model, y_pred, probs, _ = fine_tune(cv_df, test_df, tokenizer, args, device)
    y_true = test_df.label.map(LABEL2ID).values
    pd.DataFrame([{"model": MODEL_ALIAS, **compute_metrics(y_true, y_pred)}]).round(4).to_csv(f"{OUT}/hasil_data_uji.csv", index=False)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4, output_dict=True)
    pd.DataFrame(report).T.round(4).to_csv(f"{OUT}/classification_report_uji.csv")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=LABELS).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix {MODEL_ALIAS} (data uji)")
    plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

    salah = test_df.assign(prediksi=[ID2LABEL[i] for i in y_pred], keyakinan=probs.max(1).round(3))
    salah[salah.label != salah.prediksi][["tweet", "label", "prediksi", "keyakinan"]].to_csv(
        f"{OUT}/salah_prediksi.csv", index=False)
    model.save_pretrained(f"{OUT}/model_akhir"); tokenizer.save_pretrained(f"{OUT}/model_akhir")
    return model, y_pred

# 4. FAKTOR BERPENGARUH (OCCLUSION)
@torch.no_grad()
def predict_texts(model, tokenizer, texts, device, max_len, bs=32):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], max_length=max_len, padding=True, truncation=True,
                        return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu())
    return torch.cat(out).numpy()


def influential_words(model, tokenizer, test_df, y_pred, device, args, n_per_class=100, min_freq=3):
    tables = {}
    df = test_df.assign(pred=y_pred)
    for label in LABELS:
        rows = df[(df.label == label) & (df.pred == LABEL2ID[label])].head(n_per_class)
        scores = {}
        for text in rows.text:
            words = text.split()
            if len(words) < 2:
                continue
            base_p = predict_texts(model, tokenizer, [text], device, args.max_len)[0][LABEL2ID[label]]
            variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
            drops = base_p - predict_texts(model, tokenizer, variants, device, args.max_len)[:, LABEL2ID[label]]
            for w, d in zip(words, drops):
                scores.setdefault(w, []).append(d)
        tables[label] = (pd.DataFrame([(w, np.mean(v), len(v)) for w, v in scores.items() if len(v) >= min_freq],
                                      columns=["kata", "rata2_penurunan_prob", "frekuensi"])
                         .sort_values("rata2_penurunan_prob", ascending=False).head(15).round(4))
        print(f"\nKata paling berpengaruh untuk '{label}':")
        print(tables[label].head(10).to_string(index=False))
    pd.concat({k: v.reset_index(drop=True) for k, v in tables.items()}, axis=1).to_csv(
        f"{OUT}/faktor_berpengaruh.csv", index=False)

    uji = ["aku senang banget hari ini", "aku tidak senang dengan pelayanan ini",
           "kesel banget sama macet tiap pagi", "takut banget besok hasil tesnya keluar",
           "kangen kamu, cepat pulang ya", "sedih rasanya ditinggal teman dekat",
           "senang sih, tapi takut juga kalau gagal"]
    probs = predict_texts(model, tokenizer, [clean_text(t) for t in uji], device, args.max_len)
    pd.DataFrame({"kalimat": uji, "prediksi": [ID2LABEL[int(p.argmax())] for p in probs],
                  **{f"p_{l}": probs[:, i].round(3) for i, l in enumerate(LABELS)}}).to_csv(
        f"{OUT}/uji_konteks.csv", index=False)


# 5. ALUR UTAMA
def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | model: {MODEL_NAME} | folds: {args.folds} | epoch: {args.epochs} | "
          f"batch: {args.batch} | lr: {args.lr} | max_len: {args.max_len}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cv_df, test_df = load_data()
    describe_data(cv_df, test_df, tokenizer, args.max_len)

    base = baseline_cv(cv_df, args)
    base.round(4).to_csv(f"{OUT}/baseline_per_fold.csv", index=False)
    bert = cross_validate(cv_df, tokenizer, args, device)
    summarize(bert, base, args)

    if not args.skip_final:
        model, y_pred = final_test(cv_df, test_df, tokenizer, args, device)
        influential_words(model, tokenizer, test_df, y_pred, device, args)


if __name__ == "__main__":
    main()