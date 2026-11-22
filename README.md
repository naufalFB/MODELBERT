# Klasifikasi Emosi Tweet Bahasa Indonesia: IndoBERT vs mBERT

Program ini melatih dan membandingkan dua model BERT untuk mengenali emosi tweet berbahasa Indonesia. Validasi memakai 10-Fold Cross Validation.

- **Dataset:** EmoT (*Emotion Twitter*) dari benchmark [IndoNLU](https://github.com/IndoNLP/indonlu), dengan 5 label: `anger`, `fear`, `happy`, `love`, `sadness`.
- **Model:** `indobenchmark/indobert-base-p1` (IndoBERT) dan `bert-base-multilingual-uncased` (mBERT).
- **Pembanding (baseline):** TF-IDF + Logistic Regression, diuji pada fold yang sama.

## Dataset

Dataset yang dipakai adalah **EmoT** dari benchmark IndoNLU:

- Repo IndoNLU: https://github.com/IndoNLP/indonlu
- Folder dataset EmoT: https://github.com/IndoNLP/indonlu/tree/master/dataset/emot_emotion-twitter

Folder `data/` di repo ini berisi salinan tiga file dari folder EmoT tersebut: `train_preprocess.csv` (3.521 tweet), `valid_preprocess.csv` (440 tweet), dan `test_preprocess.csv` (440 tweet). Jika folder `data/` kosong, skrip akan mengunduh ketiga file itu langsung dari repo IndoNLU.

## Struktur folder

```
.
├── indobert_emot_10fold.py   # fine-tuning IndoBERT + 10-fold CV + uji akhir
├── mbert_emot_10fold.py      # sama dengan file IndoBERT, hanya model dan folder output berbeda
├── bandingkan_model.py       # menggabungkan hasil kedua model jadi tabel, grafik, uji signifikansi
├── requirements.txt
├── data/                     # train/valid/test_preprocess.csv (diunduh otomatis jika belum ada)
├── outputs/
│   ├── indobert/             # hasil IndoBERT
│   ├── mbert/                # hasil mBERT
│   └── perbandingan/         # hasil bandingkan_model.py
```

## Instalasi

Butuh Python 3.10 atau yang lebih baru.

```bash
pip install -r requirements.txt
```

GPU (CUDA atau Apple MPS) dipakai otomatis jika tersedia. Tanpa GPU, program tetap berjalan di CPU, tetapi jauh lebih lambat.

## Cara menjalankan

Jalankan kedua skrip model, lalu skrip pembanding:

```bash
python indobert_emot_10fold.py
python mbert_emot_10fold.py
python bandingkan_model.py
```

`bandingkan_model.py` membaca hasil dari `outputs/indobert/` dan `outputs/mbert/`, jadi kedua skrip model harus selesai lebih dulu.

### Opsi skrip model

| Opsi | Default | Keterangan |
| --- | --- | --- |
| `--folds` | 10 | Jumlah fold cross validation |
| `--epochs` | 3 | Jumlah epoch per fold |
| `--batch` | 16 | Batch size |
| `--lr` | 2e-5 | Learning rate (AdamW) |
| `--max-len` | 128 | Panjang maksimum token |
| `--warmup` | 0.1 | Proporsi langkah warmup learning rate |
| `--seed` | 42 | Seed acak |
| `--quick` | - | Training 2 epoch saja, untuk laptop tanpa GPU |
| `--resume` | - | Melewati fold yang hasilnya sudah tersimpan di `outputs/` |
| `--skip-final` | - | Tidak melatih model akhir dan tidak menguji pada data uji |

Contoh: melanjutkan training IndoBERT yang terhenti di tengah jalan:

```bash
python indobert_emot_10fold.py --resume
```

Di CPU, satu fold butuh sekitar 40 sampai 95 menit, jadi 10 fold bisa memakan waktu lebih dari 10 jam per model. Hasil disimpan setiap kali satu fold selesai, sehingga `--resume` bisa dipakai kalau prosesnya terhenti.

## Alur program

1. Membaca data. File CSV diambil dari `data/` atau diunduh dari GitHub IndoNLU.
2. Membersihkan teks secara minimal: huruf kecil, menghapus `[USERNAME]`/`[URL]`, tautan, mention, simbol, dan emoji, serta menyingkat huruf berulang. Stopword removal dan stemming tidak dilakukan.
3. Menggabungkan data latih dan validasi menjadi data CV. Teks kosong, duplikat, dan tweet yang juga ada di data uji dihapus (3.961 menjadi 3.937 tweet).
4. Menjalankan baseline TF-IDF + Logistic Regression pada 10 fold.
5. Fine-tuning BERT pada 10 fold dengan `StratifiedKFold` (`random_state=42`).
6. Menghitung rata-rata metrik dan menjalankan *corrected resampled t-test* terhadap baseline.
7. Melatih model akhir dengan seluruh data CV, lalu mengujinya sekali pada 440 tweet data uji.
8. Mencari kata yang paling berpengaruh dengan metode *occlusion*, dan menguji model pada beberapa kalimat buatan.

## File output

Setiap skrip model menulis ke `outputs/<model>/`:

| File | Isi |
| --- | --- |
| `distribusi_label.csv` | Jumlah tweet per emosi di data CV dan data uji |
| `contoh_data.csv` | 2 contoh tweet per emosi, sebelum dan sesudah dibersihkan |
| `statistik_token.csv` | Statistik panjang token per tweet |
| `hasil_per_fold.csv` | Akurasi, precision, recall, F1 macro, dan F1 per emosi untuk setiap fold |
| `riwayat_fold_<k>.csv` | Loss dan F1 per epoch pada fold ke-k |
| `baseline_per_fold.csv` | Hasil TF-IDF + Logistic Regression per fold |
| `ringkasan_cv.csv` | Rata-rata dan simpangan baku setiap metrik (BERT dan baseline) |
| `uji_signifikansi.csv` | Hasil uji beda BERT dan baseline |
| `f1_per_fold.png` | Grafik macro F1 per fold |
| `hasil_data_uji.csv` | Metrik pada data uji |
| `classification_report_uji.csv` | Precision, recall, dan F1 per emosi pada data uji |
| `confusion_matrix.png` | Confusion matrix pada data uji |
| `salah_prediksi.csv` | Tweet data uji yang salah ditebak |
| `faktor_berpengaruh.csv` | 15 kata paling berpengaruh per emosi |
| `uji_konteks.csv` | Prediksi pada kalimat buatan |
| `model_akhir/` | Model dan tokenizer hasil training akhir (tidak ada di repo karena ukurannya di atas 100 MB; jalankan skrip untuk membuatnya) |

`bandingkan_model.py` menulis ke `outputs/perbandingan/` dalam format `.csv` dan `.md`:

| File | Isi |
| --- | --- |
| `hasil_per_pengujian` | Akurasi dan F1 kedua model serta baseline per fold, plus rata-rata dan std |
| `f1_per_emosi` | F1 per emosi (rata-rata ± std) kedua model |
| `uji_signifikansi` | Uji beda IndoBERT vs mBERT, dan keduanya vs baseline |
| `hasil_data_uji` | Metrik kedua model pada data uji |
| `panjang_token` | Panjang token per tweet untuk tokenizer tiap model |
| `f1_per_pengujian.png` | Grafik macro F1 per fold untuk ketiga model |

## Hasil

Hasil dari training 3 epoch di CPU:

| Model | Macro F1 10-fold CV | Macro F1 data uji |
| --- | --- | --- |
| IndoBERT | 0,7545 ± 0,0251 | 0,7373 |
| mBERT | 0,6378 ± 0,0301 | 0,6393 |
| TF-IDF + Logistic Regression | 0,6712 ± 0,0229 | - |

IndoBERT unggul di kesepuluh fold, dan perbedaannya dengan mBERT signifikan (p < 0,05). Emosi `sadness` paling sulit dikenali oleh kedua model.
