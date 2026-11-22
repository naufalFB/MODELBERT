 
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = "outputs/perbandingan"
MODELS = ["mbert", "indobert"]
LABELS = ["anger", "fear", "happy", "love", "sadness"]


def to_markdown(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def save(df, name, title):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    with open(f"{OUT}/{name}.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(df) + "\n")
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def corrected_ttest(a, b, folds):
    d = (a - b).dropna()
    k, var = len(d), d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + 1 / (folds - 1)) * var) if var > 0 else float("inf")
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=k - 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    hasil = {m: pd.read_csv(f"outputs/{m}/hasil_per_fold.csv").set_index("fold") for m in MODELS}
    base_path = next(p for p in ["outputs/mbert/baseline_per_fold.csv", "outputs/indobert/baseline_per_fold.csv"]
                     if os.path.exists(p))
    base = pd.read_csv(base_path).set_index("fold")
    folds = sorted(set(hasil["mbert"].index) & set(hasil["indobert"].index))
    n_folds = max(folds)

    # 1. Tabel per pengujian (format seperti "Pengujian ke-1 ... ke-10")
    rows = []
    for k in folds:
        rows.append({"Pengujian": f"Ke-{k}",
                     "Akurasi mBERT": hasil["mbert"].loc[k, "akurasi"],
                     "F1 mBERT": hasil["mbert"].loc[k, "f1_macro"],
                     "Akurasi IndoBERT": hasil["indobert"].loc[k, "akurasi"],
                     "F1 IndoBERT": hasil["indobert"].loc[k, "f1_macro"],
                     "F1 TF-IDF+LogReg": base.loc[k, "f1_macro"]})
    tabel = pd.DataFrame(rows)
    num = tabel.columns[1:]
    tabel[num] = tabel[num].astype(float)
    rata = {"Pengujian": "Rata-rata", **{c: tabel[c].mean() for c in num}}
    std = {"Pengujian": "Std", **{c: tabel[c].std() for c in num}}
    tabel = pd.concat([tabel, pd.DataFrame([rata, std])], ignore_index=True)
    tabel[num] = tabel[num].astype(float).round(4)
    save(tabel, "hasil_per_pengujian", "Hasil per pengujian (10-Fold CV)")

    # 2. F1 per emosi (rata-rata 10 fold)
    emosi = pd.DataFrame([{"Emosi": l,
                           "mBERT": f"{hasil['mbert'][f'f1_{l}'].mean():.4f} ± {hasil['mbert'][f'f1_{l}'].std():.4f}",
                           "IndoBERT": f"{hasil['indobert'][f'f1_{l}'].mean():.4f} ± {hasil['indobert'][f'f1_{l}'].std():.4f}"}
                          for l in LABELS])
    save(emosi, "f1_per_emosi", "F1 per emosi (rata-rata ± std 10 fold)")

    # 3. Uji signifikansi berpasangan
    pasangan = [("IndoBERT vs mBERT", hasil["indobert"].f1_macro, hasil["mbert"].f1_macro),
                ("mBERT vs TF-IDF+LogReg", hasil["mbert"].f1_macro, base.f1_macro),
                ("IndoBERT vs TF-IDF+LogReg", hasil["indobert"].f1_macro, base.f1_macro)]
    uji = pd.DataFrame([{"Perbandingan": nama, "Selisih F1": round(dm, 4), "t terkoreksi": round(t, 3),
                         "p-value": round(p, 5), "Signifikan (p<0,05)": "Ya" if p < 0.05 else "Tidak"}
                        for nama, a, b in pasangan for dm, t, p in [corrected_ttest(a, b, n_folds)]])
    save(uji, "uji_signifikansi", "Uji signifikansi (corrected resampled t-test)")

    # 4. Hasil data uji akhir dan panjang token (jika sudah ada)
    uji_akhir = [pd.read_csv(f"outputs/{m}/hasil_data_uji.csv") for m in MODELS
                 if os.path.exists(f"outputs/{m}/hasil_data_uji.csv")]
    if uji_akhir:
        akhir = pd.concat(uji_akhir, ignore_index=True)[["model", "akurasi", "precision_macro", "recall_macro", "f1_macro"]]
        save(akhir.round(4), "hasil_data_uji", "Hasil pada data uji terpisah (440 tweet)")
    tok = {m: pd.read_csv(f"outputs/{m}/statistik_token.csv", index_col=0)["panjang_token"] for m in MODELS
           if os.path.exists(f"outputs/{m}/statistik_token.csv")}
    if len(tok) == 2:
        token = pd.DataFrame([{"Model": m, "Median token": s["50%"], "Persentil-95": s["95%"],
                               "Maksimum": s["max"], "% terpotong": s["persen_terpotong"]} for m, s in tok.items()])
        save(token, "panjang_token", "Panjang token per tweet (tokenizer tiap model)")

    # 5. Grafik macro F1 per pengujian
    x = np.arange(len(folds))
    plt.figure(figsize=(9, 4))
    plt.bar(x - 0.27, hasil["mbert"].loc[folds, "f1_macro"], 0.27, label="mBERT")
    plt.bar(x, hasil["indobert"].loc[folds, "f1_macro"], 0.27, label="IndoBERT")
    plt.bar(x + 0.27, base.loc[folds, "f1_macro"], 0.27, label="TF-IDF + LogReg", color="tab:gray")
    plt.xticks(x, [f"ke-{k}" for k in folds])
    plt.ylabel("macro F1"); plt.title("Macro F1 per pengujian: mBERT vs IndoBERT"); plt.legend()
    plt.savefig(f"{OUT}/f1_per_pengujian.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSemua tabel (.csv dan .md) dan grafik tersimpan di {OUT}/")


if __name__ == "__main__":
    main()