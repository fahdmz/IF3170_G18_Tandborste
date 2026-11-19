import warnings
from collections import OrderedDict
from numbers import Integral, Real

import numpy as np

from src.models.base import BaseClassifier

_KERNELS = ("linear", "poly", "rbf")
_KERNEL_ALIASES = {"polynomial": "poly", "gaussian": "rbf"}
_TAU = 1e-12
# Batas memori matriks kernel penuh (byte). Di atas ini kolom kernel dihitung saat dibutuhkan dan di-cache.
_FULL_KERNEL_BYTES = 512 * 1024**2
_CACHE_COLUMNS = 2048


def _resolve_kernel(kernel):
    name = _KERNEL_ALIASES.get(kernel, kernel)
    if name not in _KERNELS:
        raise ValueError(f"kernel harus salah satu dari {_KERNELS + tuple(_KERNEL_ALIASES)}, didapat {kernel!r}")
    return name


def _is_number(value, kind):
    return isinstance(value, kind) and not isinstance(value, bool)


def _kernel_matrix(A, B, kernel, gamma, degree, coef0, A_sq=None, B_sq=None):
    # Semua operasi in-place pada satu buffer agar memori puncak ~ satu matriks (len(A), len(B)).
    K = A @ B.T
    if kernel == "linear":
        return K
    if kernel == "poly":
        K *= gamma
        K += coef0
        return np.power(K, degree, out=K)
    A_sq = (A**2).sum(axis=1) if A_sq is None else A_sq
    B_sq = (B**2).sum(axis=1) if B_sq is None else B_sq
    K *= -2.0
    K += A_sq[:, None]
    K += B_sq[None, :]
    np.maximum(K, 0.0, out=K)
    K *= -gamma
    return np.exp(K, out=K)


def _kernel_diagonal(X, kernel, gamma, degree, coef0):
    sq_norms = (X**2).sum(axis=1)
    if kernel == "linear":
        return sq_norms
    if kernel == "poly":
        return (gamma * sq_norms + coef0) ** degree
    return np.ones(len(X))


def _validate_class_weight(class_weight):
    if class_weight is not None and class_weight != "balanced" and not isinstance(class_weight, dict):
        raise ValueError(f"class_weight harus None, 'balanced', atau dict, didapat {class_weight!r}")


def _sample_weights(class_weight, classes, y):
    """Bobot per sampel dari class_weight (None, 'balanced', atau dict label -> bobot)."""
    if class_weight is None:
        weights = {label: 1.0 for label in classes}
    elif class_weight == "balanced":
        weights = {label: len(y) / (2.0 * np.sum(y == label)) for label in classes}
    else:
        weights = {label: float(class_weight.get(label, 1.0)) for label in classes}
    return np.array([weights[label] for label in y], dtype=float)


class _KernelColumns:
    """Menyediakan kolom K[:, i] untuk SMO, baik dari matriks penuh maupun cache LRU."""

    def __init__(self, X, kernel, gamma, degree, coef0):
        self.X = X
        self.params = (kernel, gamma, degree, coef0)
        self.sq_norms = (X**2).sum(axis=1)
        n = len(X)
        self.full = None
        self.cache = OrderedDict()
        if n * n * 8 <= _FULL_KERNEL_BYTES:
            self.full = _kernel_matrix(X, X, *self.params, self.sq_norms, self.sq_norms)

    def __getitem__(self, i):
        if self.full is not None:
            return self.full[:, i]
        column = self.cache.get(i)
        if column is None:
            column = _kernel_matrix(
                self.X, self.X[i : i + 1], *self.params, self.sq_norms, self.sq_norms[i : i + 1]
            )[:, 0]
            self.cache[i] = column
            if len(self.cache) > _CACHE_COLUMNS:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(i)
        return column


class SVM(BaseClassifier):
    """SVM biner dengan solver SMO pada bentuk dual.

    Pemilihan working set memakai informasi orde kedua (WSS3) dan update dua variabel
    mengikuti LIBSVM, sehingga hasilnya dapat dibandingkan langsung dengan sklearn SVC.

    `max_iter` adalah jumlah maksimum pass, satu pass = n update pasangan SMO
    (n = jumlah sampel latih). Nilai -1 berarti tanpa batas.
    """

    # Referensi:
    # - J. Platt, "Sequential Minimal Optimization: A Fast Algorithm for Training Support
    #   Vector Machines", Microsoft Research Technical Report MSR-TR-98-14, 1998.
    # - R.-E. Fan, P.-H. Chen, C.-J. Lin, "Working Set Selection Using Second Order
    #   Information for Training Support Vector Machines", JMLR 6, 2005.
    # - C.-C. Chang, C.-J. Lin, "LIBSVM: A Library for Support Vector Machines", ACM TIST, 2011.

    def __init__(
        self,
        kernel="linear",
        C=1.0,
        degree=3,
        gamma="scale",
        coef0=1.0,
        tol=1e-3,
        max_iter=1000,
        max_train_samples=None,
        batch_size=1024,
        class_weight=None,
        random_state=42,
    ):
        super().__init__(random_state)
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size
        self.class_weight = class_weight

    def _validate_params(self):
        _resolve_kernel(self.kernel)
        if not _is_number(self.C, Real) or self.C <= 0:
            raise ValueError(f"C harus bilangan > 0, didapat {self.C!r}")
        if not _is_number(self.degree, Integral) or self.degree < 1:
            raise ValueError(f"degree harus integer >= 1, didapat {self.degree!r}")
        if isinstance(self.gamma, str):
            if self.gamma not in ("scale", "auto"):
                raise ValueError(f"gamma harus 'scale', 'auto', atau bilangan > 0, didapat {self.gamma!r}")
        elif not _is_number(self.gamma, Real) or self.gamma <= 0:
            raise ValueError(f"gamma harus 'scale', 'auto', atau bilangan > 0, didapat {self.gamma!r}")
        if not _is_number(self.tol, Real) or self.tol <= 0:
            raise ValueError(f"tol harus bilangan > 0, didapat {self.tol!r}")
        if not _is_number(self.max_iter, Integral) or (self.max_iter < 1 and self.max_iter != -1):
            raise ValueError(f"max_iter harus integer >= 1 atau -1, didapat {self.max_iter!r}")
        if self.max_train_samples is not None and (
            not _is_number(self.max_train_samples, Integral) or self.max_train_samples < 2
        ):
            raise ValueError(f"max_train_samples harus None atau integer >= 2, didapat {self.max_train_samples!r}")
        if not _is_number(self.batch_size, Integral) or self.batch_size < 1:
            raise ValueError(f"batch_size harus integer >= 1, didapat {self.batch_size!r}")
        _validate_class_weight(self.class_weight)

    def _subsample(self, X, y):
        """Subsampling terstratifikasi: proporsi kelas dipertahankan, total tepat max_train_samples."""
        n = len(y)
        total = self.max_train_samples
        if total is None or total >= n:
            return X, y
        rng = self._rng()
        neg_idx = np.flatnonzero(y == self.classes_[0])
        pos_idx = np.flatnonzero(y == self.classes_[1])
        # Minimal satu sampel per kelas; sisa kuota diberikan ke kelas lain (total < n, jadi selalu muat).
        n_pos = min(max(1, int(round(total * len(pos_idx) / n))), len(pos_idx), total - 1)
        n_neg = min(total - n_pos, len(neg_idx))
        n_pos = total - n_neg
        chosen = np.sort(
            np.concatenate(
                [rng.choice(neg_idx, size=n_neg, replace=False), rng.choice(pos_idx, size=n_pos, replace=False)]
            )
        )
        return X[chosen], y[chosen]

    def _sample_C(self, y):
        """C per sampel: C dikali bobot kelas sampel tersebut."""
        return self.C * _sample_weights(self.class_weight, self.classes_, y)

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            variance = X.var()
            return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _smo(self, K, diag, y, C):
        """Menyelesaikan min 0.5 a'Qa - e'a, 0 <= a_i <= C_i, y'a = 0, dengan Q_ij = y_i y_j K_ij."""
        n = len(y)
        alpha = np.zeros(n)
        grad = -np.ones(n)
        limit = np.inf if self.max_iter == -1 else self.max_iter * n
        n_iter = 0
        converged = False
        while n_iter < limit:
            below_upper = alpha < C
            above_lower = alpha > 0
            up = (below_upper & (y > 0)) | (above_lower & (y < 0))
            low = (below_upper & (y < 0)) | (above_lower & (y > 0))
            score = -y * grad

            # i: pelanggar KKT terbesar di himpunan "up"
            up_scores = np.where(up, score, -np.inf)
            i = int(np.argmax(up_scores))
            g_max = up_scores[i]
            g_min = np.min(np.where(low, score, np.inf))
            if g_max - g_min < self.tol:
                converged = True
                break

            # j: penurunan objektif terbesar memakai pendekatan orde kedua (WSS3)
            K_i = K[i]
            b = g_max - score
            a = diag[i] + diag - 2.0 * K_i
            a = np.where(a > 0, a, _TAU)
            candidates = low & (b > 0)
            gains = np.where(candidates, -(b * b) / a, np.inf)
            j = int(np.argmin(gains))
            K_j = K[j]

            old_i, old_j = alpha[i], alpha[j]
            quad = max(diag[i] + diag[j] - 2.0 * K_i[j], _TAU)
            if y[i] != y[j]:
                delta = (-grad[i] - grad[j]) / quad
                diff = old_i - old_j
                ai, aj = old_i + delta, old_j + delta
                if diff > 0:
                    if aj < 0:
                        aj, ai = 0.0, diff
                elif ai < 0:
                    ai, aj = 0.0, -diff
                if diff > C[i] - C[j]:
                    if ai > C[i]:
                        ai, aj = C[i], C[i] - diff
                elif aj > C[j]:
                    aj, ai = C[j], C[j] + diff
            else:
                delta = (grad[i] - grad[j]) / quad
                total = old_i + old_j
                ai, aj = old_i - delta, old_j + delta
                if total > C[i]:
                    if ai > C[i]:
                        ai, aj = C[i], total - C[i]
                elif aj < 0:
                    aj, ai = 0.0, total
                if total > C[j]:
                    if aj > C[j]:
                        aj, ai = C[j], total - C[j]
                elif ai < 0:
                    ai, aj = 0.0, total
            alpha[i], alpha[j] = ai, aj

            # G += Q[:, i] * delta_i + Q[:, j] * delta_j
            grad += y * (y[i] * (ai - old_i) * K_i + y[j] * (aj - old_j) * K_j)
            n_iter += 1

        if not converged:
            warnings.warn(
                f"SMO belum konvergen setelah {n_iter} update; naikkan max_iter atau tol",
                RuntimeWarning,
                stacklevel=3,
            )
        return alpha, grad, n_iter, converged

    @staticmethod
    def _intercept(alpha, grad, y, C):
        """Intercept b = -rho, dihitung dari kondisi KKT seperti LIBSVM."""
        y_grad = y * grad
        free = (alpha > 0) & (alpha < C)
        if np.any(free):
            rho = y_grad[free].mean()
        else:
            at_upper = alpha >= C
            at_lower = ~at_upper
            upper_mask = (at_upper & (y < 0)) | (at_lower & (y > 0))
            lower_mask = (at_upper & (y > 0)) | (at_lower & (y < 0))
            ub = y_grad[upper_mask].min() if np.any(upper_mask) else np.inf
            lb = y_grad[lower_mask].max() if np.any(lower_mask) else -np.inf
            if np.isfinite(ub) and np.isfinite(lb):
                rho = (ub + lb) / 2.0
            elif np.isfinite(ub) or np.isfinite(lb):
                rho = ub if np.isfinite(ub) else lb
            else:
                rho = 0.0
        return -rho

    def fit(self, X, y):
        self._validate_params()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"X harus array 2 dimensi, didapat {X.ndim} dimensi")
        if len(X) != len(y):
            raise ValueError(f"Jumlah baris X ({len(X)}) dan y ({len(y)}) berbeda")
        if not np.all(np.isfinite(X)):
            raise ValueError("X tidak boleh berisi NaN atau inf")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"SVM membutuhkan tepat dua kelas, didapat {len(classes)}")
        self.classes_ = classes

        X, y = self._subsample(X, y)
        y_signed = np.where(y == self.classes_[1], 1.0, -1.0)
        C = self._sample_C(y)

        # Parameter kernel dibekukan saat fit agar set_params setelahnya tidak mengubah prediksi.
        self.kernel_ = kernel = _resolve_kernel(self.kernel)
        self.gamma_ = self._resolve_gamma(X)
        self.degree_ = self.degree
        self.coef0_ = float(self.coef0)
        params = (kernel, self.gamma_, self.degree_, self.coef0_)
        K = _KernelColumns(X, *params)
        diag = _kernel_diagonal(X, *params)

        alpha, grad, self.n_iter_, self.converged_ = self._smo(K, diag, y_signed, C)
        self.intercept_ = float(self._intercept(alpha, grad, y_signed, C))

        support = np.flatnonzero(alpha > 0)
        self.support_ = support
        self.support_vectors_ = X[support]
        self.dual_coef_ = alpha[support] * y_signed[support]
        self.n_support_ = np.array([np.sum(y_signed[support] < 0), np.sum(y_signed[support] > 0)])
        if kernel == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_
        else:
            # coef_ hanya terdefinisi untuk kernel linear; buang sisa fit sebelumnya.
            vars(self).pop("coef_", None)
        return self

    def decision_function(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.support_vectors_.shape[1]:
            raise ValueError(f"X harus berbentuk (n, {self.support_vectors_.shape[1]})")
        if self.kernel_ == "linear":
            return X @ self.coef_ + self.intercept_
        scores = np.empty(len(X))
        for start in range(0, len(X), self.batch_size):
            batch = X[start : start + self.batch_size]
            K = _kernel_matrix(batch, self.support_vectors_, self.kernel_, self.gamma_, self.degree_, self.coef0_)
            scores[start : start + self.batch_size] = K @ self.dual_coef_ + self.intercept_
        return scores

    def predict(self, X):
        self._check_is_fitted()
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])


class PegasosSVM(BaseClassifier):
    """SVM linear yang dilatih di bentuk primal dengan Pegasos (stochastic sub-gradient, mini-batch).

    Meminimalkan lam/2 ||w||^2 + (1/n) sum_i c_i max(0, 1 - y_i (w.x_i + b)).
    Bias dimasukkan sebagai fitur konstan 1 sehingga ikut teregularisasi (salah satu opsi bias yang dibahas paper),
    agar langkah awal eta = 1/(lam t) yang besar tidak membuat bias berosilasi.
    Padanan dengan SVM dual: lam = 1 / (C * n).

    Tidak memerlukan matriks kernel, sehingga dapat dilatih pada seluruh data dalam O(n_epochs * n * d).
    """

    # Referensi:
    # - S. Shalev-Shwartz, Y. Singer, N. Srebro, A. Cotter, "Pegasos: Primal Estimated sub-GrAdient
    #   SOlver for SVM", Mathematical Programming 127(1), 2011, hlm. 3-30.

    def __init__(
        self,
        lam=1e-4,
        n_epochs=20,
        batch_size=64,
        project=True,
        average=True,
        class_weight=None,
        random_state=42,
    ):
        super().__init__(random_state)
        self.lam = lam
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.project = project
        self.average = average
        self.class_weight = class_weight

    def _validate_params(self):
        if not _is_number(self.lam, Real) or self.lam <= 0:
            raise ValueError(f"lam harus bilangan > 0, didapat {self.lam!r}")
        if not _is_number(self.n_epochs, Integral) or self.n_epochs < 1:
            raise ValueError(f"n_epochs harus integer >= 1, didapat {self.n_epochs!r}")
        if not _is_number(self.batch_size, Integral) or self.batch_size < 1:
            raise ValueError(f"batch_size harus integer >= 1, didapat {self.batch_size!r}")
        _validate_class_weight(self.class_weight)

    def _objective(self, w, X_aug, y_signed, weights):
        margins = y_signed * (X_aug @ w)
        return 0.5 * self.lam * (w @ w) + np.mean(weights * np.maximum(0.0, 1.0 - margins))

    def fit(self, X, y):
        self._validate_params()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"X harus array 2 dimensi, didapat {X.ndim} dimensi")
        if len(X) != len(y):
            raise ValueError(f"Jumlah baris X ({len(X)}) dan y ({len(y)}) berbeda")
        if not np.all(np.isfinite(X)):
            raise ValueError("X tidak boleh berisi NaN atau inf")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"SVM membutuhkan tepat dua kelas, didapat {len(classes)}")
        self.classes_ = classes

        n = len(y)
        X_aug = np.hstack([X, np.ones((n, 1))])
        y_signed = np.where(y == classes[1], 1.0, -1.0)
        weights = _sample_weights(self.class_weight, classes, y)
        rng = self._rng()
        k = min(self.batch_size, n)
        radius = np.sqrt(weights.max() / self.lam)

        w = np.zeros(X_aug.shape[1])
        w_sum = np.zeros_like(w)
        n_avg = 0
        t = 0
        history = []
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, k):
                batch = order[start : start + k]
                t += 1
                eta = 1.0 / (self.lam * t)
                Xb, yb, cb = X_aug[batch], y_signed[batch], weights[batch]
                violated = yb * (Xb @ w) < 1.0
                w *= 1.0 - eta * self.lam
                if np.any(violated):
                    w += (eta / len(batch)) * ((cb[violated] * yb[violated]) @ Xb[violated])
                if self.project:
                    # Proyeksi ke bola ||w|| <= sqrt(max_c / lam), tempat solusi optimal pasti berada.
                    norm = np.sqrt(w @ w)
                    if norm > radius:
                        w *= radius / norm
                # Rata-rata iterate pada paruh kedua training (suffix averaging) mengurangi noise SGD.
                if self.average and epoch >= self.n_epochs // 2:
                    w_sum += w
                    n_avg += 1
            history.append(self._objective(w, X_aug, y_signed, weights))

        if self.average and n_avg > 0:
            w = w_sum / n_avg
        self.coef_ = w[:-1].copy()
        self.intercept_ = float(w[-1])
        self.n_iter_ = t
        self.objective_history_ = np.array(history)
        self.objective_ = float(self._objective(w, X_aug, y_signed, weights))
        return self

    def decision_function(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.coef_):
            raise ValueError(f"X harus berbentuk (n, {len(self.coef_)})")
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        self._check_is_fitted()
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])
