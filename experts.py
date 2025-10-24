import os
import math
import logging
from datetime import datetime

import pandas as pd
import nibabel as nib
import numpy as np
import argparse
import math
import logging
import subprocess
import shutil
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from ViT3D import VisionTransformer3D

import os, re, shutil
from glob import glob

REGION_HINTS = {
    "GM":  ["_gm_", "gm", "gray", "grey"],
    "WM":  ["_wm_", "wm", "white"],
    "CSF": ["_csf_", "csf", "vent"]
}

def _normalize_path(p):
    if p is None: return None
    s = str(p).strip()
    return os.path.expanduser(os.path.expandvars(s))

def _maybe_alt_extension(p):
    if not isinstance(p, str): return p
    if os.path.isfile(p): return p
    if p.endswith(".nii.gz"):
        alt = p[:-3]
        if os.path.isfile(alt): return alt
    elif p.endswith(".nii"):
        alt = p + ".gz"
        if os.path.isfile(alt): return alt
    return p

def _file_readable(p):
    return isinstance(p, str) and os.path.isfile(p) and os.access(p, os.R_OK)

def _find_similar_mask_in_dir(dirpath, region):
    """Search same directory for a file that likely is the region mask."""
    hints = REGION_HINTS.get(region, [])
    patterns = []
    for h in hints:
        patterns += [
            f"*{h}*aligned_2mm.nii.gz", f"*{h}*aligned_2mm.nii",
            f"*{h}*.nii.gz",           f"*{h}*.nii",
        ]
    cands = []
    for pat in patterns:
        cands.extend(glob(os.path.join(dirpath, pat)))
    if not cands:
        return None
    # rank by how closely the name matches our hints and preferred tokens
    def score(p):
        name = os.path.basename(p).lower()
        s = 0
        for i, h in enumerate(hints):
            if h.strip("_").lower() in name: s += (10 - i)
        if "aligned_2mm" in name: s += 5
        if "6dof" in name:        s += 2
        return s
    cands = sorted(set(cands), key=score, reverse=True)
    return cands[0]

def _resolve_mask_path(mask_path, t1_path, region):
    """Return a readable path for the region mask, if we can find one."""
    p = _normalize_path(mask_path)
    p = _maybe_alt_extension(p)
    if _file_readable(p):
        return p
    # try same dir as T1 with fuzzy region hints
    t1p = _normalize_path(t1_path)
    base = os.path.dirname(t1p) if t1p else ""
    if base:
        alt = _find_similar_mask_in_dir(base, region)
        if alt and _file_readable(alt):
            return alt
    return p  # unresolved (will be filtered/logged later)

import os

def _normalize_path(p):
    if p is None:
        return None
    s = str(p).strip()
    return os.path.expanduser(os.path.expandvars(s))

def _maybe_alt_extension(p):
    # try .nii <-> .nii.gz if the given one isn't present
    if not isinstance(p, str):
        return p
    if os.path.isfile(p):
        return p
    if p.endswith(".nii.gz"):
        alt = p[:-3]            # drop .gz
        if os.path.isfile(alt):
            return alt
    elif p.endswith(".nii"):
        alt = p + ".gz"
        if os.path.isfile(alt):
            return alt
    return p

def _file_readable(p):
    return isinstance(p, str) and os.path.isfile(p) and os.access(p, os.R_OK)

def gather_region_data(df, t1_col, mask_col, age_col, region_name, is_master=True, log_dir="logs"):
    """
    Build lists of (T1 path, MASK path, AGE) for a region, silently ignoring rows
    where either file is missing/unreadable or age is NaN.
    Returns: t1_paths, mask_paths, ages, kept_count
    """
    os.makedirs(log_dir, exist_ok=True)

    # keep only rows with non-null paths and age present
    sub = df.loc[df[mask_col].notna() & df[t1_col].notna(), [t1_col, mask_col, age_col]].copy()
    if sub.empty:
        if is_master:
            print(f"[{region_name}] no rows with non-null {t1_col} & {mask_col}")
        return [], [], [], 0

    # normalize paths and try .nii <-> .nii.gz swap
    sub[t1_col]   = sub[t1_col].astype(str).map(_normalize_path).map(_maybe_alt_extension)
    sub[mask_col] = sub[mask_col].astype(str).map(_normalize_path).map(_maybe_alt_extension)

    # coerce age to numeric, drop NaNs
    sub[age_col] = pd.to_numeric(sub[age_col], errors="coerce")
    sub = sub[sub[age_col].notna()]
    if sub.empty:
        if is_master:
            print(f"[{region_name}] no rows with numeric '{age_col}' after coercion")
        return [], [], [], 0

    # keep only rows with readable files
    ok_t1   = sub[t1_col].map(_file_readable)
    ok_mask = sub[mask_col].map(_file_readable)
    ok = ok_t1 & ok_mask

    total = len(sub)
    kept  = int(ok.sum())

    # optional: log bad rows for inspection
    if is_master and kept < total:
        bad = sub.loc[~ok, [t1_col, mask_col, age_col]]
        bad.to_csv(os.path.join(log_dir, f"missing_{region_name}.csv"), index=False)
        print(f"[{region_name}] ignored {total-kept}/{total} rows (missing files). "
              f"Logged to {log_dir}/missing_{region_name}.csv")

    sub = sub.loc[ok]

    t1_paths   = sub[t1_col].tolist()
    mask_paths = sub[mask_col].tolist()
    ages       = sub[age_col].astype(float).tolist()

    if is_master:
        print(f"[{region_name}] total kept: {kept}")

    return t1_paths, mask_paths, ages, kept


EXCEL_PATH = "/ifs/loni/faculty/thompson/four_d/ADNI/Spreadsheets_ADNI/ADNI_all_T1_DLpaths_DWIpaths_demographics_20240418_shared.xlsx"
REGIONS = {
    "GM":  "NONACCEL_DL_6DOF_2MM_GM",
    "WM":  "NONACCEL_DL_6DOF_2MM_WM",
    "CSF": "NONACCEL_DL_6DOF_2MM_CSF",
}

T1_COL   = "NONACCEL_DL_6DOF_2MM_T1"
age_col  = "AGE_at_scan"

IMG_SIZE   = (91, 109, 91)
PATCH_SIZE = (7, 7, 7)
EMB_DIM    = 256
DEPTH      = 4
NUM_HEADS  = 4


LOG_PER_BATCH   = False
SAVE_SNAPSHOTS  = False

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

'''
Multi-GPU detection and utilization
'''
def mem_get_info(device_idx):
    try:
        return torch.cuda.mem_get_info(device_idx)
    except Exception:
        try:
            from torch.cuda import memory
            return memory.mem_get_info(device_idx)
        except Exception:
            cur = torch.cuda.current_device() if torch.cuda.is_available() else None
            torch.cuda.set_device(device_idx)
            try:
                free, total = torch.cuda.mem_get_info()
            finally:
                if cur is not None:
                    torch.cuda.set_device(cur)
            return free, total
def _query_gpus_via_smi():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output([
              "nvidia-smi",
           "--query-gpu=index,memory.free,memory.total",
               "--format=csv,noheader,nounits"
        ], stderr=subprocess.STDOUT, text=True)
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            idx, free_mib, total_mib = int(parts[0]), int(parts[1]), int(parts[2])
            rows.append((idx, free_mib / 1024.0, total_mib / 1024.0))
        return rows
    except Exception:
        return None

def select_gpus(min_free_gb=6.0, max_gpus=None):
    if not torch.cuda.is_available():
        return []
    info = _query_gpus_via_smi()
    if not info:
        n = torch.cuda.device_count()
        idxs = list(range(n))
        return idxs[:max_gpus] if max_gpus else idxs
    candidates = [(i, free_gb) for (i, free_gb, _tot) in info if free_gb >= float(min_free_gb)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    idxs = [i for i, _ in candidates]
    if max_gpus is not None and max_gpus > 0:
        idxs = idxs[:max_gpus]
    return idxs

def is_ddp_run():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ

def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def setup_ddp(backend="nccl"):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)  # torchrun sets env vars for us
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

'''
Main code start here
'''

class ExpertRegressor3D(nn.Module):
    def __init__(self, vit3d: VisionTransformer3D):
        super().__init__()
        self.vit = vit3d
        self.head = nn.Linear(vit3d.embed_dim, 1)

    def forward(self, x):
        tokens = self.vit(x)
        if isinstance(tokens, tuple):  # in case ViT3D returns (tokens, attn)
            tokens = tokens[0]
        cls    = tokens[:,0,:]
        out    = self.head(cls).squeeze(-1)
        return out

class BrainRegionDataset(Dataset):
    def __init__(self, t1_paths, mask_paths, ages):
        self.t1_paths   = t1_paths
        self.mask_paths = mask_paths
        self.ages       = ages

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        t1_img   = nib.load(self.t1_paths[idx])
        mask_img = nib.load(self.mask_paths[idx])

        t1   = t1_img.get_fdata().astype(np.float32)
        mask = (mask_img.get_fdata() > 0.5).astype(np.float32)

        region = t1 * mask

        m = region[mask > 0].mean() if (mask > 0).any() else 0.0
        s = region[mask > 0].std()  if (mask > 0).any() else 1.0
        if s > 1e-8:
            region = (region - m) / s

        region_tensor = torch.from_numpy(region).unsqueeze(0)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        return region_tensor, age


def make_logger(region: str) -> logging.Logger:
    logger = logging.getLogger(f"train_{region}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join("logs", f"train_log_{region}.txt"))
        fmt = logging.Formatter("%(asctime)s\t%(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def build_loader(dataset, batch_size, ddp=False, seed=42, drop_last=False):
    if ddp and get_world_size() > 1:
        sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=drop_last)
        g = torch.Generator()
        g.manual_seed(seed + get_rank())
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, generator=g)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    return loader

def _require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. "
                       f"Available: {list(df.columns)}")

def load_checkpoint(model, region, device, ckpt_dir="checkpoints", strict=True, logger=None):
    ckpt_path = os.path.join(ckpt_dir, f"expert_{region}.pth")
    if not os.path.isfile(ckpt_path):
        if logger: logger.info(f"[{region}] no pretrained weights found at {ckpt_path}; training from scratch.")
        else:      print(f"[{region}] no pretrained weights found at {ckpt_path}; training from scratch.")
        return False, ckpt_path

    try:
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state, strict=strict)
        except RuntimeError:
            # Handle DP/DDP "module." prefix mismatch
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                new_state[k.replace("module.", "")] = v
            model.load_state_dict(new_state, strict=False)
        if logger: logger.info(f"[{region}] loaded pretrained weights: {ckpt_path}")
        else:      print(f"[{region}] loaded pretrained weights: {ckpt_path}")
        return True, ckpt_path
    except Exception as e:
        if logger: logger.info(f"[{region}] failed to load {ckpt_path}: {e}; training from scratch.")
        else:      print(f"[{region}] failed to load {ckpt_path}: {e}; training from scratch.")
        return False, ckpt_path


def train_region(region: str, df: pd.DataFrame, args, device, ddp=False):
    mask_col = REGIONS[region]

    for c in (T1_COL, mask_col, age_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in Excel.")

    is_master = (get_rank() == 0)

    t1_paths, mask_paths, ages, kept = gather_region_data(
        df, T1_COL, mask_col, age_col,
        region_name=region, is_master=is_master, log_dir="logs"
    )
    if kept == 0:
        if is_master:
            print(f"[{region}] 0 usable rows after filtering; skipping this expert.")
        return

    # if mask_col is None:
    #     raise KeyError(f"Unknown region '{region}'. Expected one of {list(REGIONS.keys())}")
    #
    # cols_needed = [T1_COL, mask_col]
    # if age_col is not None:
    #     cols_needed.append(age_col)
    # _require_cols(df, cols_needed)
    #
    # sub = df[df[mask_col].notna()].copy()
    # if sub.empty:
    #     raise RuntimeError(f"No rows found with non-null mask for column '{mask_col}'.")
    #
    # sub[age_col] = pd.to_numeric(sub[age_col], errors="coerce")
    # sub = sub[sub[age_col].notna()]
    #
    # t1_paths   = sub[T1_COL].tolist()
    # mask_paths = sub[mask_col].tolist()
    # ages       = sub[age_col].astype(float).tolist()
    #
    # sub[T1_COL] = sub[T1_COL].astype(str).map(_normalize_path).map(_maybe_alt_extension)
    # sub[mask_col] = sub[mask_col].astype(str).map(_normalize_path).map(_maybe_alt_extension)
    #
    # # try to resolve missing masks using the T1 folder + region hints
    # resolved = []
    # for mp, tp in zip(sub[mask_col].tolist(), sub[T1_COL].tolist()):
    #     resolved.append(_resolve_mask_path(mp, tp, region))
    # sub[mask_col] = resolved
    #
    # # verify files now
    # ok_t1 = sub[T1_COL].map(_file_readable)
    # ok_mask = sub[mask_col].map(_file_readable)
    # ok = ok_t1 & ok_mask
    # bad = sub.loc[~ok, [T1_COL, mask_col, age_col]]
    #
    # is_master = (get_rank() == 0)
    # if not ok.all() and is_master:
    #     os.makedirs("logs", exist_ok=True)
    #     bad.to_csv(os.path.join("logs", f"missing_{region}.csv"), index=False)
    #     print(f"[warn] {region}: {len(bad)} rows missing/unreadable files; logged to logs/missing_{region}.csv")
    #
    # sub = sub.loc[ok].copy()
    # if sub.empty:
    #     raise RuntimeError(
    #         f"All rows invalid for {region} after path verification. "
    #         f"Likely your spreadsheet mask filenames don’t match disk. "
    #         f"See logs/missing_{region}.csv"
    #     )

    dataset   = BrainRegionDataset(t1_paths, mask_paths, ages)
    loader    = build_loader(dataset, args.batch_size, ddp=ddp)

    vit  = VisionTransformer3D(img_size=args.img_size, patch_size=args.patch_size,
                               in_chans=1, embed_dim=args.emb_dim,
                               depth=args.depth, num_heads=args.num_heads).to(device)
    model = ExpertRegressor3D(vit).to(device)

    if ddp and get_world_size() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    is_master = (get_rank() == 0)
    logger = make_logger(region)

    logger.info(f"Start training {region}: N={len(dataset)}, "
                f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
                f"dist={'DDP' if ddp and get_world_size()>1 else ('DP' if len(args.device_ids)>1 else 'single')}")
    _ = load_checkpoint(model, region, device, ckpt_dir="checkpoints", strict=True, logger=logger)

    for epoch in range(args.epochs):
        if ddp and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)  # ensure fresh shuffling in DDP each epoch. :contentReference[oaicite:7]{index=7}

        model.train()
        running = 0.0
        n_batches = 0
        pbar = loader if not is_master else tqdm(loader, desc=f"[{region}] epoch {epoch+1}/{args.epochs}", leave=False)
        for i, (X, y) in enumerate(pbar):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(X)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            loss  = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

            if LOG_PER_BATCH and is_master:
                logger.info(f"epoch={epoch+1} batch={i+1}/{len(loader)} loss={loss.item():.6f}")
            if is_master and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running / max(1, n_batches)
        if is_master:
            logger.info(f"epoch={epoch+1} avg_loss={epoch_loss:.6f}")

            latest_path = os.path.join("checkpoints", f"expert_{region}.pth")
            to_save = model.module.state_dict() if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model.state_dict()
            torch.save(to_save, latest_path)

            if SAVE_SNAPSHOTS:
                snap = os.path.join("checkpoints", f"expert_{region}_epoch{epoch+1:03d}.pth")
                torch.save(to_save, snap)

    if is_master:
        logger.info(f"Finished training {region}.")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=EXCEL_PATH, type=str)
    ap.add_argument("--batch_size", default=10, type=int)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)

    ap.add_argument("--img_size", default="91,109,91", type=str)
    ap.add_argument("--patch_size", default="7,7,7", type=str)
    ap.add_argument("--emb_dim", default=EMB_DIM, type=int)
    ap.add_argument("--depth", default=DEPTH, type=int)
    ap.add_argument("--num_heads", default=NUM_HEADS, type=int)

    ap.add_argument("--dist", choices=["ddp", "dp", "none"], default="dp",
                    help="ddp (recommended with torchrun), dp (DataParallel), or none")
    ap.add_argument("--min_free_gb", default=6.0, type=float, help="only use GPUs with at least this much free GB")
    ap.add_argument("--max_gpus", default=8, type=int, help="cap number of GPUs used")

    return ap.parse_args()

def main():
    args = parse_args()

    args.img_size = tuple(int(x) for x in args.img_size.split(","))
    args.patch_size = tuple(int(x) for x in args.patch_size.split(","))

    sel = select_gpus(min_free_gb=args.min_free_gb, max_gpus=args.max_gpus)
    if not sel:
        print("No suitable GPUs found (min_free_gb {}). Using CPU (VERY slow).".format(args.min_free_gb))
        device = torch.device("cpu")
        args.device_ids = []
        ddp = False
    else:
        if args.dist == "ddp" and is_ddp_run():
            setup_ddp(
                backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            args.device_ids = [local_rank]
            ddp = True
        elif args.dist == "ddp" and not is_ddp_run():
            print("[warn] --dist ddp requested but not launched with torchrun; falling back to DataParallel.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in sel)
            device = torch.device("cuda:0")
            args.device_ids = list(range(len(sel)))
            ddp = False
        elif args.dist == "dp":
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in sel)
            device = torch.device("cuda:0")
            args.device_ids = list(range(len(sel)))
            ddp = False
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(sel[0])
            device = torch.device("cuda:0")
            args.device_ids = [0]
            ddp = False

    df = pd.read_excel(args.excel)
    
    # ---- Train each region expert
    try:
        grand_total = 0
        for region in ["GM", "WM", "CSF"]:
            kept_before = grand_total
            train_region(region, df, args, device, ddp=ddp)

        print(f"[ALL] grand total kept: {grand_total}")
    finally:
        if ddp:
            cleanup_ddp()

if __name__ == "__main__":
    main()