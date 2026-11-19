# Catatan SVM (Anggota 5)

## Konfigurasi terbaik untuk notebook

```python
from src.models.svm import SVM

svm = SVM(kernel="rbf", C=3.0, gamma=0.1, max_train_samples=None, max_iter=-1)
# pipeline: make_numeric_pipeline(num, cat) tanpa PCA, tanpa resampler, threshold default (predict)
```

- Macro F1 5-fold CV 0.821 ± 0.009, ROC AUC 0.897, fit sekitar 30–60 detik per fold (21.8k baris).
- Ganti `max_train_samples=5000` di notebook ke `None`. Dengan train penuh, Macro F1 naik sekitar +0.03 (0.790 → 0.821).
- Threshold tidak perlu disetel untuk Macro F1. Threshold optimal di skor `decision_function` sekitar −0.13 dan hanya menambah sekitar +0.002.
- Untuk cost-sensitive (FP:FN = 2:1), `class_weight="balanced"` punya AUC tertinggi (0.900). Threshold bisa dipilih dari skor `decision_function` dengan `tune_threshold`.

## Arti parameter yang perlu diketahui

- `max_iter`: jumlah maksimum *pass*, dengan 1 pass = n update pasangan SMO (n = jumlah sampel latih). `-1` berarti tanpa batas. Default 1000 sudah lebih dari cukup: di data ini SMO konvergen sekitar 26k update, sedangkan batasnya 1000 × 21.8k. Kalau berhenti sebelum konvergen, muncul `RuntimeWarning` dan `converged_ = False`.
- `max_train_samples`: subsampling terstratifikasi (memakai `random_state`) sebelum training. `None` berarti pakai semua data.
- `kernel`: `"linear"`, `"poly"`/`"polynomial"`, `"rbf"`/`"gaussian"`. `coef0` default 1.0 (berbeda dari default sklearn, yaitu 0.0).
- `class_weight`: `None`, `"balanced"`, atau dict `{label: bobot}`. Bobotnya mengalikan C per sampel.
- Atribut hasil fit: `support_`, `support_vectors_`, `dual_coef_` (α·y), `intercept_`, `n_support_`, `n_iter_`, `converged_`, `kernel_`, `gamma_`, `degree_`, `coef0_`. `coef_` hanya ada untuk kernel linear.

## Hasil perbandingan (5-fold, baris latih identik)

Sumber: `output/svm_comparison/svm_scratch_vs_sklearn_summary.csv`, dibuat ulang dengan `python -m src.experiments.svm_experiments` (fold NumPy sendiri; verifikasi dengan modul tim di bagian bawah).

| Konfigurasi | Macro F1 scratch / sklearn | AUC | Fit scratch / sklearn (dtk) | Prediksi sama |
| :--- | :--- | :--- | :--- | :--- |
| rbf C=3 γ=0.1 (terbaik) | 0.8209 / 0.8209 | 0.897 | 59.8 / 58.1 | 100% |
| rbf C=1 γ=scale | 0.7956 / 0.7955 | 0.886 | 26.8 / 30.4 | 99.99% |
| poly d=3 C=1 | 0.7963 / 0.7963 | 0.888 | 69.7 / 27.9 | 100% |
| linear C=0.1, 5000 sampel | 0.7559 / 0.7558 | 0.853 | 9.0 / 1.7 | 99.97% |
| Pegasos λ=1e-3 balanced vs `LinearSVC` | 0.7532 / 0.7528 | 0.856 | 1.8 / 0.4 | 99.57% |

Catatan:
- Waktu training diukur saat 5 konfigurasi dilatih paralel, tapi kondisinya sama untuk scratch dan sklearn.
- Pegasos memakai threshold 0. Dengan threshold yang disetel, Macro F1-nya 0.769 (`output/svm_comparison/pegasos_tuning_cv.csv`).
- Kernel linear (SMO maupun Pegasos) tertinggal sekitar 0.05–0.07 Macro F1 dari rbf. Data ini butuh batas keputusan non-linear.

## Riwayat tuning

`output/svm_comparison/svm_tuning_cv.csv` berisi 67 konfigurasi yang dievaluasi bertahap: grid kasar dengan 5000 sampel, lalu 10k sampel, lalu train penuh. Kolom `macro_f1_tuned_thr` memakai threshold yang dipilih dari skor out-of-fold fold lain, sehingga tidak bocor. File ini dibuat dengan fold dari `sklearn.model_selection.StratifiedKFold`, sedangkan modul `src/experiments/svm_experiments.py` memakai fold NumPy sendiri. Karena itu angkanya bisa beda sekitar ±0.001. Untuk mengulang grid dengan fold modul: `python -m src.experiments.svm_experiments --tune`.

Temuan utama:
- Jumlah sampel latih paling berpengaruh. Untuk rbf C=3: 5k → 10k → penuh = 0.790 → 0.804 → 0.816–0.822.
- rbf > poly > linear. Puncaknya di C=3, γ=0.1. Nilai γ ≥ 0.2 atau C ≥ 5 mulai menurunkan skor.
- PCA mempercepat training sekitar 2× pada 5000 sampel, dengan Macro F1 hampir sama. Tidak dipakai untuk konfigurasi akhir.

## Bonus

- Plot 2D bidang pemisah, margin f(x)=±1, dan support vector per kernel: `src/visualization/svm_boundary.py` (`plot_svm_kernels`). Gambarnya ada di `output/figures/svm_kernels_2d_synthetic.png` (data sintetis) dan `output/figures/svm_kernels_2d_pca.png` (data asli, 2 komponen PCA).
- Algoritma optimasi di luar materi kelas: `PegasosSVM` (primal stochastic sub-gradient). Belum dicek di salindia bahwa Pegasos belum diajarkan. Referensinya ada di `docs/referensi_sementara.md`.

## Verifikasi dengan modul evaluasi tim

`output/svm_comparison/svm_team_cv_comparison.csv` dibuat dengan `StratifiedKFold` dan `cross_validate` dari `src/evaluation` (pipeline di-fit per fold), dijalankan lewat `python -m src.experiments.svm_experiments --team-cv`:

| Konfigurasi | Impl | Macro F1 | F1 kelas batal | Recall | Akurasi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| rbf C=3 γ=0.1 | scratch | 0.8209 ± 0.008 | 0.7527 | 0.7116 | 0.8469 |
| rbf C=3 γ=0.1 | sklearn | 0.8209 ± 0.008 | 0.7528 | 0.7118 | 0.8469 |
| rbf C=1 γ=scale | scratch | 0.7956 ± 0.005 | 0.7127 | 0.6469 | 0.8292 |
| rbf C=1 γ=scale | sklearn | 0.7955 ± 0.005 | 0.7125 | 0.6467 | 0.8291 |

`cross_validate` tim hanya meneruskan label ke metrik, jadi ROC AUC dan pencarian threshold (butuh skor kontinu) tetap dihitung oleh `src/experiments/svm_experiments.py` dengan fold NumPy sendiri.

## Belum dikerjakan

- `predict_proba` (Platt scaling) tidak diimplementasikan. Untuk threshold dan ROC AUC cukup pakai `decision_function`.
- Belum dicek di salindia bahwa Pegasos belum diajarkan (syarat bonus).
