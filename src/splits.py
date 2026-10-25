"""Random and spatially-disjoint block splits, with formal leakage checks.

Split maps use the codes 0 = not used, 1 = train, 2 = val, 3 = test.

Spatial split
-------------
1. Tile the scene into ``block x block`` squares (grid origin at pixel 0,0; edge blocks are smaller).
2. Assign every block wholly to train / val / test. We draw ``n_trials`` random assignments
   (probabilities 0.6 / 0.2 / 0.2, seeded) and keep the one with the lowest score, where the
   score is the deviation of the realised labelled-pixel shares from 60/20/20 overall and per
   class, plus a large penalty for every class with >= ``min_class_px`` pixels that is missing from
   test (or val / train). Only ground-truth *counts* are used - never model performance.
3. Buffer: a CNN input patch of radius ``r`` centred on a training pixel must not contain any test
   pixel. We drop training (and validation) pixels whose Chebyshev distance to any pixel of a
   test block is <= r. We also drop training pixels within r of a validation block so that
   early stopping / tuning is not optimistic. Buffers are measured to the *block area*, which is
   stricter than measuring to labelled test pixels only.
4. ``verify_spatial`` re-checks all of this from the final masks (used by the tests and asserted at
   run time).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

TRAIN, VAL, TEST = 1, 2, 3


def random_split(gt: np.ndarray, train_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Stratified: ``train_frac`` of every class (at least 1 pixel) to train, the rest to test."""
    rng = np.random.default_rng(seed)
    split = np.zeros(gt.shape, np.int8)
    flat = split.ravel()
    for c in np.unique(gt[gt > 0]):
        idx = np.flatnonzero(gt.ravel() == c)
        rng.shuffle(idx)
        n_tr = max(1, int(round(train_frac * idx.size)))
        flat[idx[:n_tr]] = TRAIN
        flat[idx[n_tr:]] = TEST
    return split


def block_ids(shape, block: int) -> np.ndarray:
    H, W = shape
    rows = np.arange(H)[:, None] // block
    cols = np.arange(W)[None, :] // block
    return (rows * ((W + block - 1) // block) + cols).astype(np.int64)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))


def apply_buffer(block_assign: np.ndarray, gt: np.ndarray, r: int) -> np.ndarray:
    """block_assign: (H, W) split code of the block each pixel lies in. Returns labelled split map."""
    test_area = block_assign == TEST
    val_area = block_assign == VAL
    near_test = _dilate(test_area, r)
    near_val = _dilate(val_area, r)
    lab = gt > 0
    split = np.zeros(gt.shape, np.int8)
    split[lab & test_area] = TEST
    split[lab & val_area & ~near_test] = VAL
    split[lab & (block_assign == TRAIN) & ~near_test & ~near_val] = TRAIN
    return split


def _score(split, gt, n_cls, min_class_px, targets):
    counts = np.zeros((4, n_cls + 1))
    np.add.at(counts, (split.ravel(), gt.ravel()), 1)
    counts = counts[1:, 1:]                          # (3 splits, K classes)
    tot = counts.sum()
    s = 3.0 * sum(abs(counts[i].sum() / tot - targets[i]) for i in range(3))
    cls_tot = np.bincount(gt.ravel(), minlength=n_cls + 1)[1:]
    big = cls_tot >= min_class_px
    with np.errstate(invalid="ignore", divide="ignore"):
        share = counts / np.maximum(counts.sum(0, keepdims=True), 1)
    s += 0.5 * np.mean([abs(share[i, big] - targets[i]).mean() for i in range(3)])
    missing_test = int(np.sum(big & (counts[2] == 0)))
    missing_other = int(np.sum(big & ((counts[0] == 0) | (counts[1] == 0))))
    return s + 10.0 * missing_test + 0.15 * missing_other, missing_test


def spatial_split(gt: np.ndarray, block: int = 20, seed: int = 0, buffer: int = 3,
                  targets=(0.6, 0.2, 0.2), n_trials: int = 1500, min_class_px: int = 50):
    """Returns (split_map, block_assign_map, info dict)."""
    rng = np.random.default_rng(seed)
    bid = block_ids(gt.shape, block)
    n_blocks = bid.max() + 1
    lab_per_block = np.bincount(bid.ravel(), weights=(gt.ravel() > 0), minlength=n_blocks)
    has_lab = lab_per_block > 0
    n_cls = int(gt.max())
    best = None
    for _ in range(n_trials):
        codes = np.zeros(n_blocks, np.int8)  # blocks without labels stay 0 (unused)
        # the buffer removes pixels from train only, so over-sample train blocks by a random delta
        d = rng.uniform(0, 0.15)
        p = np.array([targets[0] + d, targets[1] - d / 2, targets[2] - d / 2])
        codes[has_lab] = rng.choice([TRAIN, VAL, TEST], size=has_lab.sum(), p=p / p.sum())
        assign = codes[bid]
        split = apply_buffer(assign, gt, buffer)
        sc, miss = _score(split, gt, n_cls, min_class_px, targets)
        if best is None or sc < best[0]:
            best = (sc, split, assign, miss)
    sc, split, assign, miss = best
    info = split_summary(split, gt)
    info.update(block=block, buffer=buffer, seed=seed, n_trials=n_trials, score=float(sc),
                classes_ge_min_missing_from_test=int(miss), n_blocks=int(n_blocks),
                n_blocks_labelled=int(has_lab.sum()),
                labelled_lost_to_buffer=int(((gt > 0) & (split == 0)).sum()))
    return split, assign, info


def split_summary(split: np.ndarray, gt: np.ndarray) -> dict:
    n_cls = int(gt.max())
    out = {}
    for code, name in [(TRAIN, "train"), (VAL, "val"), (TEST, "test")]:
        m = split == code
        out[f"n_{name}"] = int(m.sum())
        out[f"class_counts_{name}"] = np.bincount(gt[m], minlength=n_cls + 1)[1:].tolist()
    tot = sum(out[f"n_{k}"] for k in ("train", "val", "test"))
    for k in ("train", "val", "test"):
        out[f"frac_{k}"] = out[f"n_{k}"] / max(tot, 1)
    out["n_labelled_total"] = int((gt > 0).sum())
    return out


def verify_spatial(split: np.ndarray, assign: np.ndarray, gt: np.ndarray, block: int, r: int,
                   min_class_px: int = 50) -> dict:
    """Formal checks. Raises AssertionError on any violation; returns a report."""
    bid = block_ids(gt.shape, block)
    # (a) every block has a single split code
    for b in np.unique(bid):
        codes = np.unique(assign[bid == b])
        assert codes.size == 1, f"block {b} has mixed assignment {codes}"
    # (b) labelled pixels only take the code of their own block (or 0 when buffered out)
    lab = split > 0
    assert np.all(split[lab] == assign[lab]), "pixel split differs from its block assignment"
    # (c) no training patch (radius r) contains a test-block pixel
    tr = split == TRAIN
    test_area = assign == TEST
    assert not np.any(_dilate(tr, r) & test_area), "training patch touches a test block"
    assert not np.any(_dilate(tr, r) & (assign == VAL)), "training patch touches a validation block"
    assert not np.any(_dilate(split == VAL, r) & test_area), "validation patch touches a test block"
    # (d) class coverage in test
    tot = np.bincount(gt.ravel(), minlength=gt.max() + 1)
    in_test = np.bincount(gt[split == TEST], minlength=gt.max() + 1)
    missing = [int(c) for c in range(1, gt.max() + 1) if tot[c] >= min_class_px and in_test[c] == 0]
    assert not missing, f"classes with >= {min_class_px} px missing from test: {missing}"
    return {"blocks_pure": True, "train_patch_clear_of_test": True, "val_patch_clear_of_test": True,
            "classes_missing_from_test": missing}


def verify_random(split: np.ndarray, gt: np.ndarray, frac: float = 0.1) -> dict:
    lab = gt > 0
    assert np.all(split[lab] > 0) and np.all(split[~lab] == 0)
    for c in np.unique(gt[lab]):
        n = (gt == c).sum()
        n_tr = ((gt == c) & (split == TRAIN)).sum()
        assert n_tr == max(1, int(round(frac * n))), (c, n_tr, n)
    return {"stratified": True}
