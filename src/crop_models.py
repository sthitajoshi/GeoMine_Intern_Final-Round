"""Classical (RBF-SVM) and deep (small 3-D spectral-spatial CNN) crop classifiers.

Both are tuned on the validation pixels only; test pixels are touched once, after the model
and its hyper-parameters are frozen.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC

# ----------------------------------------------------------------------------- SVM


def tune_svm(Xtr, ytr, Xva, yva, grid_C=(1, 10, 100, 1000), grid_gamma=(0.001, 0.005, 0.02),
             seed=0, max_train=None):
    """Grid search on the validation set (OA). Returns best params and the full table."""
    Xtr, ytr = _cap(Xtr, ytr, max_train, seed)
    table = []
    for C, g in itertools.product(grid_C, grid_gamma):
        m = SVC(C=C, gamma=g, kernel="rbf", random_state=seed).fit(Xtr, ytr)
        table.append({"C": C, "gamma": g, "val_oa": float((m.predict(Xva) == yva).mean())})
    best = max(table, key=lambda d: d["val_oa"])
    return {"C": best["C"], "gamma": best["gamma"]}, table


def fit_svm(Xtr, ytr, C, gamma, seed=0, max_train=None):
    """Final SVM with Platt-scaled probabilities (5-fold internal CV on the training set)."""
    Xtr, ytr = _cap(Xtr, ytr, max_train, seed)
    return SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=seed).fit(Xtr, ytr)


def svm_proba(model: SVC, X, n_classes):
    """predict_proba with columns expanded to all classes (absent training classes get 0)."""
    P = np.zeros((len(X), n_classes), np.float32)
    for i in range(0, len(X), 20000):
        P[i:i + 20000][:, model.classes_] = model.predict_proba(X[i:i + 20000])
    return P


def _cap(X, y, max_n, seed):
    """Stratified subsample to at most max_n pixels (SVM training is O(n^2) memory/time)."""
    if not max_n or len(y) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, int(round(max_n * idx.size / len(y))))
        keep.append(rng.choice(idx, min(k, idx.size), replace=False))
    keep = np.concatenate(keep)
    return X[keep], y[keep]


# ----------------------------------------------------------------------------- CNN

class CNN3D(nn.Module):
    """Small 3-D CNN on (1, bands, p, p) patches; three 3-D convs then two FC layers.

    With p = 7 the three valid 3x3 spatial convolutions reduce the patch to 1x1, so the
    receptive field is exactly the input patch (radius 3) - which is what the split buffer
    guarantees is clear of test pixels.
    """

    def __init__(self, n_bands: int, n_classes: int, patch: int = 7, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3)), nn.BatchNorm3d(8), nn.ReLU(),
            nn.Conv3d(8, 16, (5, 3, 3)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, (3, 3, 3)), nn.BatchNorm3d(32), nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, n_bands, patch, patch)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def extract_patches(Z: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch: int) -> np.ndarray:
    """Z: (H, W, C) preprocessed cube -> (N, 1, C, p, p). Reflect padding at scene edges
    (reflection only re-uses pixels already within the radius, so it adds no new neighbours)."""
    r = patch // 2
    Zp = np.pad(Z, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.empty((len(rows), 1, Z.shape[2], patch, patch), np.float32)
    for k, (i, j) in enumerate(zip(rows, cols)):
        out[k, 0] = Zp[i:i + patch, j:j + patch, :].transpose(2, 0, 1)
    return out


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_cnn(Ztr_patches, ytr, Zva_patches, yva, n_classes, seed=0, lr=1e-3, epochs=80,
              batch=128, patience=12, weight_decay=1e-4, dropout=0.4, augment=True, log=None):
    """Adam + cross-entropy; checkpoint/early stopping on validation OA (val loss breaks ties). Returns (model, history)."""
    _set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = CNN3D(Ztr_patches.shape[2], n_classes, Ztr_patches.shape[-1], dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt = torch.from_numpy(Ztr_patches), torch.from_numpy(ytr.astype(np.int64))
    Xv, yv = torch.from_numpy(Zva_patches), torch.from_numpy(yva.astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = -np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=g)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            if augment:  # random flips / 90-degree rotations of the patch (label-preserving)
                if torch.rand(1, generator=g) < 0.5:
                    xb = xb.flip(-1)
                xb = torch.rot90(xb, int(torch.randint(0, 4, (1,), generator=g)), dims=(-2, -1))
            loss = nn.functional.cross_entropy(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vl, va = _eval_loss(model, Xv, yv)
        hist.append({"epoch": ep, "train_loss": tot / len(yt), "val_loss": vl, "val_oa": va})
        score = va - 1e-3 * vl            # select on validation OA; loss only breaks ties
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    if log:
        log(f"    cnn lr={lr} stopped ep={len(hist)} best_val_score={best:.3f} ({time.time()-t0:.0f}s)")
    return model, hist


@torch.no_grad()
def _eval_loss(model, X, y, batch=1024):
    model.eval()
    tot, corr = 0.0, 0
    for i in range(0, len(y), batch):
        out = model(X[i:i + batch])
        tot += nn.functional.cross_entropy(out, y[i:i + batch], reduction="sum").item()
        corr += (out.argmax(1) == y[i:i + batch]).sum().item()
    return tot / len(y), corr / len(y)


@torch.no_grad()
def cnn_logits(model, patches: np.ndarray, batch=2048) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(patches[i:i + batch])).numpy() for i in range(0, len(patches), batch)]
    return np.concatenate(out)


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits_val, y_val) -> float:
    """Temperature scaling (Guo et al. 2017) fitted on validation logits by 1-D grid search on NLL."""
    best_T, best_nll = 1.0, np.inf
    for T in np.exp(np.linspace(np.log(0.25), np.log(8), 120)):
        p = softmax(logits_val, T)
        nll = -np.mean(np.log(p[np.arange(len(y_val)), y_val] + 1e-12))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T
