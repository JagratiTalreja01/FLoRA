# data/dataset_sen1floods11.py
# Sen1Floods11 loader — robust pairing, dB-aware SAR scaling, dtype-aware S2 scaling

import os, re, glob, random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

# ---------- IO ----------

def _read_any(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()  # C,H,W
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- Scaling ----------

def _to_float01_s2(x: np.ndarray) -> np.ndarray:
    """
    Sentinel-2 bands → [0,1] with dtype/range awareness.
    Works for uint8, uint16, or float arrays already in reflectance-ish range.
    """
    x = x.astype(np.float32, copy=False)
    if x.dtype == np.uint16:
        x = x / 10000.0
    elif x.dtype == np.uint8:
        x = x / 255.0
    else:
        vmax = float(np.max(x)) if x.size else 0.0
        if vmax > 2000.0:
            x = x / 10000.0
        elif vmax > 1.5:
            x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def _robust_minmax(x: np.ndarray, p_lo=2.0, p_hi=98.0) -> np.ndarray:
    """Percentile min–max (robust to outliers & NaNs)."""
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    bx = x[finite]
    if bx.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(bx, p_lo), np.percentile(bx, p_hi)
    if hi <= lo:
        lo, hi = float(bx.min()), float(bx.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)

def _scale_band_to_01(b: np.ndarray) -> np.ndarray:
    """
    dB-aware SAR scaler:
      • If the band looks like dB (median < 0 and max <= ~5), DO NOT clamp negatives
        and DO NOT log again → just robust-rescale the raw values.
      • Otherwise assume amplitude/power → clamp negatives to 0, log1p, robust-rescale.
    This avoids collapsing typical Sen1 dB tiles to a constant (the 'all blue' bug).
    """
    b = b.astype(np.float32, copy=False)
    finite = np.isfinite(b)
    bx = b[finite]
    if bx.size == 0:
        return np.zeros_like(b, dtype=np.float32)

    looks_db = (np.median(bx) < 0.0) and (bx.max() <= 5.0)
    if looks_db:
        out = _robust_minmax(b, 2.0, 98.0)
    else:
        b_lin = b.copy()
        b_lin[b_lin < 0] = 0
        b_log = np.log1p(b_lin)
        out = _robust_minmax(b_log, 2.0, 98.0)
    return out

def _scale_sar_to_01(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    vv01 = _scale_band_to_01(vv)
    vh01 = _scale_band_to_01(vh)
    return np.stack([vv01, vh01], axis=0)

def _compute_ndvi_or_gray(s2_full: np.ndarray) -> np.ndarray:
    C,H,W = s2_full.shape
    if C >= 8:
        # Typical S2 L2A indices: B04 (red)=3, B08 (NIR)=7
        red = s2_full[3].astype(np.float32) + 1e-6
        nir = s2_full[7].astype(np.float32) + 1e-6
        ndvi = (nir - red) / (nir + red)
        return np.clip(ndvi, -1.0, 1.0)[None, ...]
    if C >= 3:
        r,g,b = s2_full[0], s2_full[1], s2_full[2]
        gray = (0.2989*r + 0.5870*g + 0.1140*b).astype(np.float32)[None, ...]
        return _to_float01_s2(gray)
    return np.zeros((1,H,W), dtype=np.float32)

# ---------- Pairing keys ----------

_STEM_CLEAN_RE = re.compile(r'(\.tif+|\.tiff+|\.png|\.jpg|\.jpeg)$', re.IGNORECASE)

def _stem_key(p: str) -> str:
    b = os.path.basename(p); b = os.path.splitext(b)[0]
    # strip vv/vh tokens and common separators conservatively
    b = re.sub(r'(?i)\bvv\b|\bvh\b|_vv|_vh|_s1|_s2', '', b)
    b = re.sub(r'[\s\-_.]+', '_', b)
    return b.strip('_').lower()

def _dir1_key(p: str) -> str:
    return os.path.basename(os.path.dirname(p)).strip().lower()

def _dir2_key(p: str) -> str:
    d1 = os.path.dirname(p)
    return os.path.basename(os.path.dirname(d1)).strip().lower()

def _dir12_key(p: str) -> str:
    return f"{_dir2_key(p)}/{_dir1_key(p)}".lower()

def _choose_key_fn(mode: str):
    if mode == 'stem':  return _stem_key
    if mode == 'dir1':  return _dir1_key
    if mode == 'dir2':  return _dir2_key
    if mode == 'dir12': return _dir12_key
    raise ValueError(f"pair_key '{mode}' not in ['stem','dir1','dir2','dir12']")

# ---------- Scan S1 with stack support ----------

def _scan_s1(root_s1: str, key_fn) -> Dict[str, Dict[str,str]]:
    """Return dict: key -> {'vv': path, 'vh': path} or {'stack': path} if 2-band tif."""
    out: Dict[str, Dict[str,str]] = {}
    paths = glob.glob(os.path.join(root_s1, '**', '*.*'), recursive=True)
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            continue
        key = key_fn(p)
        name = os.path.basename(p).lower()

        # Detect a 2-band stack quickly
        is_stack = False
        try:
            with rasterio.open(p) as ds:
                if ds.count >= 2 and ext in ('.tif', '.tiff'):
                    is_stack = True
        except Exception:
            pass

        if is_stack:
            out[key] = {'stack': p}
            continue

        # Otherwise, separate vv/vh by filename hints
        if 'vv' in name:
            out.setdefault(key, {})['vv'] = p
        elif 'vh' in name:
            out.setdefault(key, {})['vh'] = p
        else:
            out.setdefault(key, {}).setdefault('other', p)

    # prune keys without usable SAR
    clean = {}
    for k, rec in out.items():
        if 'stack' in rec or ('vv' in rec and 'vh' in rec):
            clean[k] = rec
    return clean

# ---------- Scan S2 ----------

def _scan_s2(root_s2: str, key_fn) -> Dict[str, str]:
    out = {}
    for p in glob.glob(os.path.join(root_s2, '**', '*.*'), recursive=True):
        if os.path.splitext(p)[1].lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            out[key_fn(p)] = p
    return out

def _find_mask_for_s2(s2_path: str) -> Optional[str]:
    base = os.path.splitext(s2_path)[0]
    candidates = [
        base.replace('S2Hand', 'LabelHand') + '.tif',
        base.replace('S2Hand', 'S2Label') + '.tif',
        base + '_label.tif',
        base + '_mask.tif',
    ]
    for c in candidates:
        if os.path.exists(c): return c
    root = os.path.dirname(os.path.dirname(s2_path))
    hits = glob.glob(os.path.join(root, '**', '*mask*.tif'), recursive=True)
    return hits[0] if hits else None

# ---------- Build pairs with fallback order ----------

def _build_pairs_with_fallback(root_s1: str, root_s2: str, pair_key: str):
    """Try requested key; if 0 pairs, auto-fallback through sensible order."""
    orders = {
        'stem':  ['stem', 'dir1', 'dir2', 'dir12'],
        'dir1':  ['dir1', 'stem', 'dir2', 'dir12'],
        'dir2':  ['dir2', 'dir1', 'stem', 'dir12'],
        'dir12': ['dir12','dir1','dir2','stem'],
        'auto':  ['stem', 'dir1', 'dir2', 'dir12'],
    }
    tried = []
    for mode in orders.get(pair_key, orders['auto']):
        tried.append(mode)
        key_fn = _choose_key_fn(mode)
        s1 = _scan_s1(root_s1, key_fn)
        s2 = _scan_s2(root_s2, key_fn)
        pairs = []
        for k, rec in s1.items():
            if k not in s2: continue
            if 'stack' in rec:
                p_vv = f"{rec['stack']}::stack::0"
                p_vh = f"{rec['stack']}::stack::1"
            else:
                if 'vv' not in rec or 'vh' not in rec: continue
                p_vv, p_vh = rec['vv'], rec['vh']
            p_s2 = s2[k]
            p_mask = _find_mask_for_s2(p_s2)
            pairs.append((p_vv, p_vh, p_s2, p_mask))
        if len(pairs) > 0:
            return pairs, mode, tried
    return [], None, tried

# ---------- Dataset ----------

class Sen1Floods11Dataset(Dataset):
    """
    Args:
      root_s1, root_s2: paths to S1Hand / S2Hand
      split: 'train'|'val'|'test'
      crop_size: int
      val_ratio: float
      use_ndvi_prior: bool
      pair_key: 'auto'|'stem'|'dir1'|'dir2'|'dir12'
    """
    def __init__(self,
                 root_s1: str,
                 root_s2: str,
                 split: str='train',
                 crop_size: int=256,
                 val_ratio: float=0.1,
                 use_ndvi_prior: bool=True,
                 pair_key: str='auto'):
        super().__init__()
        self.root_s1 = root_s1
        self.root_s2 = root_s2
        self.split = split.lower()
        self.size  = int(crop_size)
        self.use_ndvi_prior = use_ndvi_prior

        pairs, used_mode, tried = _build_pairs_with_fallback(root_s1, root_s2, pair_key)
        if len(pairs) == 0:
            raise RuntimeError(
                "No S1<->S2 pairs found.\n"
                f"S1={root_s1}\nS2={root_s2}\n"
                f"(pair_key requested: {pair_key}; tried order: {tried})"
            )

        # deterministic split
        pairs = sorted(pairs, key=lambda x: os.path.basename(x[2]).lower())
        n = len(pairs); n_val = max(1, int(round(n * float(val_ratio))))
        if self.split == 'train':
            self.items = pairs[:-n_val] if n_val < n else pairs
        else:
            self.items = pairs[-n_val:] if n_val < n else pairs

        # debug report
        print(f"[dataset] pairing key used: {used_mode} | total pairs: {n} | "
              f"train={len(pairs[:-n_val]) if self.split=='train' else 0} | val={len(pairs[-n_val:]) if self.split!='train' else 0}")
        for j, (vv, vh, s2, m) in enumerate(self.items[:15]):
            def bn(x): 
                return os.path.basename(x.split('::')[0])
            print(f"  {j:02d} VV={bn(vv)} | VH={bn(vh)} | S2={os.path.basename(s2)} | MASK={'y' if m else 'n'}")

    def __len__(self): 
        return len(self.items)

    # crops
    def _center_crop(self, arr: np.ndarray, size: int) -> np.ndarray:
        C,H,W = arr.shape
        if H == size and W == size: return arr
        y = max(0, (H-size)//2); x = max(0, (W-size)//2)
        return arr[:, y:y+size, x:x+size]

    def _random_crop(self, arr: np.ndarray, size: int) -> np.ndarray:
        C,H,W = arr.shape
        if H <= size or W <= size:
            arr = np.pad(arr, ((0,0),(0,max(0,size-H)),(0,max(0,size-W))), mode='reflect')
            C,H,W = arr.shape
        y = random.randint(0, H-size); x = random.randint(0, W-size)
        return arr[:, y:y+size, x:x+size]

    def __getitem__(self, idx: int):
        p_vv, p_vh, p_s2, p_mask = self.items[idx]

        # S1 — handle stacks
        if "::stack::" in p_vv and "::stack::" in p_vh:
            base = p_vv.split("::stack::")[0]
            band_vv = int(p_vv.split("::stack::")[1])
            band_vh = int(p_vh.split("::stack::")[1])
            s1 = _read_any(base)  # C,H,W
            s1_vv = s1[band_vv]; s1_vh = s1[band_vh]
        else:
            s1_vv = _read_any(p_vv)[0]; s1_vh = _read_any(p_vh)[0]

        # ---- dB-aware scaling (fixes the "all-blue SAR" issue) ----
        sar = _scale_sar_to_01(s1_vv, s1_vh)

        # S2
        s2_full = _read_any(p_s2)  # C,H,W
        s2_rgb = (s2_full[:3] if s2_full.shape[0] >= 3
                  else np.repeat(s2_full[:1], 3, axis=0))
        s2_rgb = _to_float01_s2(s2_rgb.transpose(1,2,0)).transpose(2,0,1)

        # prior
        prior = _compute_ndvi_or_gray(s2_full) if self.use_ndvi_prior else None

        # mask
        if p_mask and os.path.exists(p_mask):
            m = _read_any(p_mask); m = m[0] if m.ndim == 3 else m
            mask = (m > 0.5).astype(np.float32)[None, ...]
        else:
            mask = np.zeros((1, s2_rgb.shape[1], s2_rgb.shape[2]), dtype=np.float32)

        # crops: deterministic for val/test, random+flips for train
        if self.split in ('val','validation','test'):
            sar  = self._center_crop(sar, self.size)
            s2_rgb = self._center_crop(s2_rgb, self.size)
            mask = self._center_crop(mask, self.size)
            if prior is not None: prior = self._center_crop(prior, self.size)
        else:
            sar  = self._random_crop(sar, self.size)
            s2_rgb = self._random_crop(s2_rgb, self.size)
            mask = self._random_crop(mask, self.size)
            if prior is not None: prior = self._random_crop(prior, self.size)
            if random.random() < 0.5:
                sar = sar[:, :, ::-1]; s2_rgb = s2_rgb[:, :, ::-1]; mask = mask[:, :, ::-1]
                if prior is not None: prior = prior[:, :, ::-1]
            if random.random() < 0.5:
                sar = sar[:, ::-1, :]; s2_rgb = s2_rgb[:, ::-1, :]; mask = mask[:, ::-1, :]
                if prior is not None: prior = prior[:, ::-1, :]

        sample = {'sar': torch.from_numpy(sar.copy()).float(),
                  'opt': torch.from_numpy(s2_rgb.copy()).float(),
                  'mask': torch.from_numpy(mask.copy()).float()}
        if prior is not None:
            sample['prior'] = torch.from_numpy(prior.copy()).float()
        return sample
