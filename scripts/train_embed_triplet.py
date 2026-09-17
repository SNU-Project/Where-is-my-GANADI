"""E4: E3(품종 사전학습) 백본에서, CE(BNNeck) 손실에 배치-하드 Triplet 손실을 더해 이어 학습.

배경: 실제 사진 두 건을 연속으로 테스트한 결과, 갤러리에 진짜 같은 품종·색깔의 개가 있어도
(회색 푸들 vs 회색 푸들) 임베딩 유사도가 0.09 수준으로 사실상 무작위였다 — 자세·조명·미용
스타일이 다르면 identity CE(분류) 손실만으로는 "같은 개체/비슷한 개"를 가깝게 두라는 직접적인
신호가 없다는 뜻. BNNeck 원 논문(Luo et al., "Bag of Tricks")도 원래 ID loss + Triplet loss를
함께 쓰도록 설계돼 있는데, 지금까지 이 프로젝트는 ID loss(CE)만 썼다 — 나머지 절반을 마저 적용.

Triplet loss는 임베딩 "거리"를 직접 최적화해서 (앵커-포지티브는 가깝게, 앵커-네거티브는 멀게)
분류 경계와 무관하게 진짜 유사도 구조를 학습시킨다 — 자세/조명 불변성 개선을 직접 겨냥.

배치 구성: PK 샘플링 (P개체 x 개체당 최대 K장) — src/data/episodic.py의 group_by_label 재사용.
초기화: E3 백본(checkpoints/breed_pretrained_resnet50_features.pt)에서 시작.

사용:
    python scripts/train_embed_triplet.py --steps 1500 --eval-every 100
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.episodic import group_by_label
from src.data.image_cache import load_rgb
from src.data.manifests import load_dogfacenet_split
from src.data.train_dataset import build_combined_train_set, build_mpdd_val_closed_set
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.backbones import BNNeckModel, get_device
from src.retrieval.extract import extract_features
from src.retrieval.metrics import compute_distmat, evaluate_reid

CKPT_DIR = Path("checkpoints")


class PKSampler:
    """개체(라벨)마다 최소 2장 있는 풀에서, 매 스텝 P개체 x 최대 K장을 뽑는다."""

    def __init__(self, pools: dict, p: int, k: int, seed: int = 0):
        self.pools = {l: paths for l, paths in pools.items() if len(paths) >= 2}
        self.labels = sorted(self.pools.keys())
        self.p, self.k = min(p, len(self.labels)), k
        self.rng = np.random.default_rng(seed)

    def sample(self):
        chosen = self.rng.choice(self.labels, size=self.p, replace=False)
        items = []
        for label in chosen:
            paths = self.pools[label]
            n = min(self.k, len(paths))
            picked = self.rng.choice(len(paths), size=n, replace=False)
            items += [(paths[i], label) for i in picked]
        return items


def load_batch(items, transform, device):
    imgs = torch.stack([transform(load_rgb(p)) for p, _ in items]).to(device)
    labels = torch.tensor([l for _, l in items], dtype=torch.long, device=device)
    return imgs, labels


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Hermans et al. 2017 배치-하드 triplet — 앵커마다 배치 내 가장 먼 포지티브 /
    가장 가까운 네거티브만 써서 손실을 구성 (쉬운 쌍은 무시해 학습 신호를 집중시킴)."""
    emb = F.normalize(embeddings, dim=1)
    dist = torch.cdist(emb, emb, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    dist_pos = dist.masked_fill(~pos_mask, -1.0)
    hardest_pos = dist_pos.max(dim=1).values
    dist_neg = dist.masked_fill(~neg_mask, float("inf"))
    hardest_neg = dist_neg.min(dim=1).values

    valid = pos_mask.any(dim=1)  # 배치에 포지티브가 없는(=단독 샘플) 앵커는 제외
    loss = F.relu(hardest_pos - hardest_neg + margin)[valid]
    return loss.mean(), (hardest_pos[valid] < hardest_neg[valid]).float().mean().item()


@torch.no_grad()
def evaluate_mpdd_val_closed_set(model, items, transform, device, batch_size=64):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs = torch.stack([transform(load_rgb(it.path)) for it in chunk]).to(device)
        labels = torch.tensor([it.global_label for it in chunk], device=device)
        _, logits = model.forward_with_logits(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(chunk)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_dogfacenet_val_openset(model, transform, device, batch_size):
    split = load_dogfacenet_split(group="val")
    q_feats = extract_features(split.query_paths, model, transform, device, batch_size)
    g_feats = extract_features(split.gallery_paths, model, transform, device, batch_size)
    distmat = compute_distmat(q_feats, g_feats)
    return evaluate_reid(
        distmat,
        np.asarray(split.query_pids), np.asarray(split.gallery_pids),
        np.asarray(split.query_camids), np.asarray(split.gallery_camids),
        max_rank=10,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--p", type=int, default=16, help="배치당 개체 수")
    ap.add_argument("--k", type=int, default=4, help="개체당 최대 이미지 수")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--triplet-margin", type=float, default=0.3)
    ap.add_argument("--triplet-weight", type=float, default=1.0)
    ap.add_argument("--init-backbone", default="checkpoints/breed_pretrained_resnet50_features.pt")
    ap.add_argument("--ckpt-suffix", default="breedpretrain_triplet")
    args = ap.parse_args()

    device = get_device()
    print(f"[info] device={device}")

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_ds, num_classes, label_map = build_combined_train_set(train_transform, include_mpdd=True)
    mpdd_val_ds = build_mpdd_val_closed_set(eval_transform, label_map)
    pools = group_by_label(train_ds.items)
    sampler = PKSampler(pools, p=args.p, k=args.k)
    print(f"[info] train: {len(train_ds)}장 / {num_classes}개체 (triplet 가능 풀: {len(sampler.labels)}개체), "
          f"MPDD val(closed-set): {len(mpdd_val_ds)}장")

    model = BNNeckModel(num_classes=num_classes, pretrained=True,
                         init_backbone_path=args.init_backbone).to(device)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.ckpt_suffix}" if args.ckpt_suffix else ""
    best_path = CKPT_DIR / f"E1_resnet50_bnneck{tag}.pt"
    log_csv = Path(f"metadata/train_log_E1{tag}.csv")
    log_rows = []
    best_rank1 = -1.0

    model.train()
    t0 = time.time()
    ce_sum, tri_sum, acc_sum, n = 0.0, 0.0, 0.0, 0
    for step in range(1, args.steps + 1):
        items = sampler.sample()
        imgs, labels = load_batch(items, train_transform, device)

        bn_feat, logits = model(imgs)
        ce_loss = ce_criterion(logits, labels)
        tri_loss, tri_acc = batch_hard_triplet_loss(bn_feat, labels, margin=args.triplet_margin)
        loss = ce_loss + args.triplet_weight * tri_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        ce_sum += ce_loss.item(); tri_sum += tri_loss.item(); acc_sum += tri_acc; n += 1

        if step % args.eval_every == 0 or step == args.steps:
            mpdd_acc = evaluate_mpdd_val_closed_set(model, mpdd_val_ds.items, eval_transform, device)
            dfn_metrics = evaluate_dogfacenet_val_openset(model, eval_transform, device, 32)
            elapsed = time.time() - t0
            row = {"step": step, "ce_loss": ce_sum / n, "triplet_loss": tri_sum / n,
                   "triplet_acc": acc_sum / n, "mpdd_val_acc": mpdd_acc,
                   "dfn_val_rank1": dfn_metrics.rank1, "dfn_val_map": dfn_metrics.mAP,
                   "seconds": round(elapsed, 1)}
            print(f"[step {step}/{args.steps}] ce={row['ce_loss']:.3f} triplet={row['triplet_loss']:.3f} "
                  f"triplet_acc={row['triplet_acc']:.3f} | MPDD val_acc={mpdd_acc:.3f} | "
                  f"DogFaceNet val {dfn_metrics} | {elapsed:.1f}s")
            log_rows.append(row)
            ce_sum, tri_sum, acc_sum, n = 0.0, 0.0, 0.0, 0
            t0 = time.time()
            model.train()

            if dfn_metrics.rank1 > best_rank1:
                best_rank1 = dfn_metrics.rank1
                torch.save({"model": model.state_dict(), "num_classes": num_classes,
                            "step": step, "dfn_val_rank1": best_rank1}, best_path)
                print(f"  [저장] 새 최고 DogFaceNet val Rank-1={best_rank1:.4f} -> {best_path}")

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[완료] 로그: {log_csv}, 최고 체크포인트: {best_path} (DogFaceNet val Rank-1={best_rank1:.4f})")


if __name__ == "__main__":
    main()
