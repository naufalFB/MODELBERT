# Penggunaan AI

Proyek ini dikerjakan dengan bantuan AI (Claude dari Anthropic) sebagai asisten. Saya yang menentukan kasus, dataset, metode, dan rancangan program, lalu menjalankan sendiri semua training dan pengujian di laptop. Semua angka hasil diambil langsung dari folder `outputs/` tanpa diubah.

## Riwayat percakapan dengan AI

| No | Prompt | Jawaban AI | Yang saya lakukan |
| --- | --- | --- | --- |
| 1 | Meminta rekomendasi dataset yang mudah untuk tugas data mining dengan BERT | Menyarankan tugas klasifikasi sentimen dan beberapa dataset (SmSA, IMDB, SMS Spam) | Mempertimbangkan pilihan dataset |
| 2 | Memberikan template tugas case method dan meminta analisis kasus | Menyusun draf laporan analisis sentimen | Mengganti template dengan struktur laporan yang diminta dosen |
| 3 | Meminta link setiap dataset agar bisa dicek sendiri | Memberikan 5 dataset teks berbahasa Indonesia beserta link, kolom, label, dan jumlah data | Memeriksa dataset dan memilih 10-Fold Cross Validation sebagai metode validasi |
| 4 | Membahas rancangan program: dua model (mBERT dan IndoBERT), 10-fold, dan tiga skrip terpisah | Menyusun kode sesuai rancangan tersebut | Meninjau kode, menyiapkan proyek di VS Code, dan menjalankan ketiga skrip |
| 5 | Menanyakan error dan peringatan yang muncul saat training | Menjelaskan penyebab dan cara mengatasinya | Memperbaiki masalah dan melanjutkan training sampai selesai |
| 6 | Menanyakan apakah menjalankan dua skrip sekaligus membuat lebih lama | Menyarankan menjalankan skrip satu per satu dengan opsi `--resume` | Menjalankan skrip mBERT dan IndoBERT secara berurutan |
