# Tugas Besar 2 IF3170 Inteligensi Artifisial, Kelompok 18 Tandborste

Repositori ini berisi implementasi from scratch KNN, Decision Tree (C4.5), Logistic Regression, SVM, dan Naive Bayes. Model-model tersebut digunakan untuk klasifikasi biner `is_canceled` pada kompetisi Kaggle, dengan Macro F1 sebagai metrik evaluasi.

## Setup

Gunakan Python 3.13 (versi yang dipakai saat pengembangan).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cara Menjalankan

1. Letakkan dataset kompetisi di folder `data/` pada root repo:
   - `data/train.csv`
   - `data/test.csv`
   - `data/sample_submission.csv`
2. Jalankan `main.ipynb` dari atas ke bawah.

File submission akan dihasilkan di `output/submissions/`.

## Struktur Repo

- `src/`: kode sumber (model, preprocessing, evaluasi, eksperimen, visualisasi, submission)
- `output/`: hasil eksperimen, model tersimpan, gambar, dan file submission
- `docs/`: catatan dan referensi pendukung
- `main.ipynb`: notebook utama yang menjalankan seluruh alur dari data hingga submission

## Pembagian Tugas

| Nama | NIM | Tugas |
| --- | --- | --- |
| Daniel Anindito Nugroho | 13524002 | SVM, Pegasos SVM, visualisasi bidang pemisah |
| Muhammad Raihan Iqbal | 13524011 | Logistic Regression (L1, L2, Elastic Net), Adam, animasi kontur loss |
| Fahd Muhammad Zahid | 13524078 | Setup repo, interface model, data, EDA, preprocessing, submission, integrasi notebook |
| Al Farabi | 13524086 | Decision Tree C4.5, tuning, visualisasi tree |
| Ishaq Irfan Farizal | 13524094 | Modul evaluasi, KNN, Naive Bayes, cost-sensitive learning, visualisasi graf NB |
