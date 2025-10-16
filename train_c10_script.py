import os, time, csv, random, torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from ERMoE import ERMoE

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt


class ExpertActivityTracker:
    def __init__(self, num_experts: int, save_dir: str):
        self.num_experts = num_experts
        self.save_dir = os.path.join(save_dir, "expert_activity")
        os.makedirs(self.save_dir, exist_ok=True)
        self.buf = None      # [L, E] float64 counts
        self.initialized = False

    def _layers_from_aux(self, aux):
        if aux is None:
            return []
        for k in ["routing", "routers", "moe_layers", "router_outputs", "expert_indices", "topk_per_layer"]:
            if isinstance(aux, dict) and k in aux:
                return aux[k]
        if isinstance(aux, (list, tuple)):
            return aux
        return []

    def _ensure_init(self, num_layers: int):
        if not self.initialized:
            self.buf = torch.zeros(num_layers, self.num_experts, dtype=torch.float64)
            self.initialized = True

    def update(self, aux):
        # Case A: flat dict with [L,B,T,K] top-k indices
        if isinstance(aux, dict) and "topk_indices" in aux and torch.is_tensor(aux["topk_indices"]):
            idx = aux["topk_indices"]              # [L, B, T, K]
            L = idx.size(0)
            self._ensure_init(L)
            with torch.no_grad():
                for l in range(L):
                    flat = idx[l].reshape(-1).to(torch.long).detach().cpu()
                    self.buf[l].index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
            return

        # Case B: per-layer objects (list/tuple)
        layers = self._layers_from_aux(aux)
        if not layers:
            return

        self._ensure_init(len(layers))
        with torch.no_grad():
            for l, item in enumerate(layers):
                if isinstance(item, dict):
                    if "indices" in item and torch.is_tensor(item["indices"]):
                        flat = item["indices"].reshape(-1).to(torch.long).detach().cpu()
                        self.buf[l].index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
                    elif "mask" in item and torch.is_tensor(item["mask"]):
                        # binary dispatch mask [B,T,E]
                        self.buf[l] += item["mask"].sum(dim=(0, 1)).to(torch.float64).detach().cpu()
                    elif "gates" in item and torch.is_tensor(item["gates"]):
                        self.buf[l] += item["gates"].sum(dim=(0, 1)).to(torch.float64).detach().cpu()
                elif torch.is_tensor(item):
                    if item.dim() == 3 and item.size(-1) <= self.num_experts:
                        flat = item.reshape(-1).to(torch.long).detach().cpu()
                        self.buf[l].index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
                    elif item.dim() >= 2 and item.size(-1) == self.num_experts:
                        self.buf[l] += item.sum(dim=tuple(range(item.dim()-1))).to(torch.float64).detach().cpu()

    def flush_epoch(self, epoch: int, split: str):
        if self.buf is None:
            return
        counts = self.buf.clone()
        totals = counts.sum(dim=1, keepdim=True).clamp_min_(1.0)
        pct = (counts / totals) * 100.0

        # CSV (append across epochs)
        csv_path = os.path.join(self.save_dir, f"expert_activity_{split}.csv")
        newfile = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["epoch", "layer", "expert", "count", "percent"])
            for l in range(counts.size(0)):
                for e in range(self.num_experts):
                    w.writerow([epoch, l, e, int(counts[l, e].item()), f"{pct[l, e].item():.4f}"])

        # Heatmap
        fig = plt.figure(figsize=(1.2*self.num_experts, 0.6*counts.size(0)+2))
        ax = plt.gca()
        im = ax.imshow(pct.numpy(), aspect="auto")
        plt.colorbar(im, ax=ax, label="% tokens/activations")
        ax.set_yticks(range(counts.size(0)))
        ax.set_yticklabels([f"L{l}" for l in range(counts.size(0))])
        ax.set_xticks(range(self.num_experts))
        ax.set_xticklabels([f"E{e}" for e in range(self.num_experts)], rotation=45, ha="right")
        ax.set_title(f"Expert activity per layer — {split} — epoch {epoch}")
        plt.tight_layout()
        fig_path = os.path.join(self.save_dir, f"heatmap_{split}_ep{epoch:03d}.png")
        plt.savefig(fig_path, dpi=180)
        plt.close(fig)

        # console digest
        topk = torch.topk(pct, k=min(3, self.num_experts), dim=1)
        summary = ", ".join([f"L{l}: " + "; ".join([f"E{int(e)}={v:.1f}%" for v, e in zip(topk.values[l], topk.indices[l])])
                             for l in range(pct.size(0))])
        print(f"[{split}] epoch {epoch} expert activity (top-3 per layer): {summary}")

        self.buf.zero_()


# -------------------------
# Data
# -------------------------
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=None)

def collate_pil(batch):
    imgs, labels = zip(*batch)  # imgs: list of PIL.Image, labels: tuple of int
    return list(imgs), torch.tensor(labels, dtype=torch.long)

train_ld = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_pil)
test_ld  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_pil)

# -------------------------
# Model / Opt / Logs
# -------------------------
log_dir = "./logs_ermoe"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "train_log.csv")
ckpt_best_path = os.path.join(log_dir, "ermoe_best.pth")
ckpt_last_path = os.path.join(log_dir, "ermoe_last.pth")

model = ERMoE(
    num_classes=10,
    num_experts=8,
    top_k=2,
    threshold=0.5,                 # default threshold; adjust if needed
    free_experts=(0,),
    pretrained_name="google/vit-base-patch16-224",
    device=device,
)

# optional warm-start
if os.path.exists(ckpt_best_path):
    try:
        state = torch.load(ckpt_best_path, map_location=device)
        # support both pure weights and checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Loaded weights from {ckpt_best_path}")
    except Exception as e:
        print(f"Skip loading {ckpt_best_path}: {e}")

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "split", "loss", "acc", "time_sec"])

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

trk_tr = ExpertActivityTracker(num_experts=model.num_experts, save_dir=log_dir)
trk_te = ExpertActivityTracker(num_experts=model.num_experts, save_dir=log_dir)

# -------------------------
# Train / Eval Epoch
# -------------------------
def run_epoch(loader, train_mode=True, tracker=None):
    model.train(mode=train_mode)
    # keep experts always eval to avoid dropout etc.
    for exp in model.experts:
        exp.eval()

    total_loss, total_acc, total_n = 0.0, 0.0, 0
    t0 = time.time()

    for images, labels in tqdm(loader):
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            opt.zero_grad(set_to_none=True)

        logits, aux = model(images)

        # feed a per-layer list with a dict that includes a [B, T=1, E] mask
        if tracker is not None and isinstance(aux, dict) and "sel_mask" in aux:
            mask = aux["sel_mask"]
            if torch.is_tensor(mask):
                tracker.update([{"mask": mask.unsqueeze(1)}])

        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            opt.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        total_n += bs

    dt = time.time() - t0
    return total_loss / total_n, total_acc / total_n, dt

# -------------------------
# Loop
# -------------------------
best_acc = 0.0
epochs = 100

for ep in range(1, epochs + 1):
    tr_loss, tr_acc, tr_t = run_epoch(train_ld, train_mode=True,  tracker=trk_tr)
    trk_tr.flush_epoch(ep, "train")

    te_loss, te_acc, te_t = run_epoch(test_ld,  train_mode=False, tracker=trk_te)
    trk_te.flush_epoch(ep, "test")

    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([ep, "train", f"{tr_loss:.6f}", f"{tr_acc:.4f}", f"{tr_t:.2f}"])
        w.writerow([ep, "test",  f"{te_loss:.6f}", f"{te_acc:.4f}", f"{te_t:.2f}"])

    # save "last" (full checkpoint)
    torch.save({
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "test_acc": te_acc
    }, ckpt_last_path)

    # save "best" (weights only)
    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), ckpt_best_path)
    print(f"epoch {ep:03d} | train acc {tr_acc:.4f} | test acc {te_acc:.4f} | best {best_acc:.4f}")

print("done.")
