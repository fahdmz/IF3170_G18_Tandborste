# Referensi Sementara

Daftar sumber metode dan kode yang diadaptasi, untuk dipindahkan ke laporan. Setiap anggota menambah bagiannya sendiri.

## SVM (Anggota 5)

| Dipakai di | Sumber |
| :--- | :--- |
| `src/models/svm.py` `SVM._smo` (solver dual SMO) | J. Platt, "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines", Microsoft Research Technical Report MSR-TR-98-14, 1998. |
| `SVM._smo` (pemilihan working set orde kedua, WSS3) | R.-E. Fan, P.-H. Chen, C.-J. Lin, "Working Set Selection Using Second Order Information for Training Support Vector Machines", *Journal of Machine Learning Research* 6, 2005, hlm. 1889–1918. |
| `SVM._smo` (update dua variabel dengan C per sampel) dan `SVM._intercept` (perhitungan rho dari kondisi KKT) | C.-C. Chang, C.-J. Lin, "LIBSVM: A Library for Support Vector Machines", *ACM Transactions on Intelligent Systems and Technology* 2(3), 2011. Dokumen implementasi: <https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf> (bagian penyelesaian subproblem dua variabel dan perhitungan b/ρ). |
| `SVM._resolve_gamma` (`gamma="scale"` = 1 / (n_fitur · Var(X))) | Dokumentasi scikit-learn `sklearn.svm.SVC`, parameter `gamma`: <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>. |
| `src/models/svm.py` `PegasosSVM` (bonus algoritma optimasi di luar materi kelas) | S. Shalev-Shwartz, Y. Singer, N. Srebro, A. Cotter, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", *Mathematical Programming* 127(1), 2011, hlm. 3–30. Dipakai: varian mini-batch, proyeksi opsional ke bola radius √(c_max/λ), dan bias sebagai fitur konstan (opsi penanganan bias yang dibahas di paper). Cek ulang nomor gambar/bagian saat menulis laporan. |
| `PegasosSVM` (rata-rata iterate paruh kedua / suffix averaging) | A. Rakhlin, O. Shamir, K. Sridharan, "Making Gradient Descent Optimal for Strongly Convex Stochastic Optimization", *ICML*, 2012. |
