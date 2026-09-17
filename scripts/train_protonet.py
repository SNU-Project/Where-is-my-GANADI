"""E2: Prototypical Network 에피소드 학습 (Snell et al., 2017).

선행연구(Yeon et al. 2025, docs/references/)가 DogFaceNet처럼 개체당 이미지가 극히 적은
데이터에서 메타러닝이 일반 파인튜닝을 크게 앞선다는 것을 보여 채택. E1과 초기화 조건(ImageNet
사전학습)을 동일하게 맞춰 "학습 방식 차이"만 비교되도록 한다 (E1은 CE+BNNeck, E2는 에피소드).

에피소드 구성 (N-way, 적응형 K/Q — src/data/episodic.py):
    - N-way: 매 에피소드 20개 개체 무작위 샘플링 (MPDD+DogFaceNet 통합 풀에서)
    - support: 개체당 1장 (2장짜리 개체도 참여 가능하게)
    - query: 개체당 최대 4장 (많이 가진 개체가 더 많이 기여)
거리: 임베딩을 L2 정규화한 뒤 제곱 유클리드 거리 사용 -> 정규화 공간에서는 코사인 유사도와
순위가 동일해, 프로젝트 전체에서 쓰는 cosine 기반 검색 평가와 일관성을 맞춘다.

검증은 E1과 동일한 프로토콜(DogFaceNet val, open-set Rank-1/mAP)을 그대로 재사용해 공정 비교.

사용:
    python scripts/train_protonet.py --episodes 600
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # 파일로 리다이렉트해도 진행상황이 실시간으로 보이게
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.episodic import EpisodeSampler, group_by_label
from src.data.image_cache import load_rgb
from src.data.manifests import load_dogfacenet_split
from src.data.train_dataset import build_combined_train_set
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.backbones import ProtoEmbedder, get_device
from src.retrieval.extract import extract_features
from src.retrieval.metrics import compute_distmat, evaluate_reid

CKPT_DIR = Path("checkpoints")
LOG_CSV = Path("metadata/train_log_E2.csv")


def load_batch(paths, transform, device):
    imgs = [transform(load_rgb(p)) for p in paths]
    return torch.stack(imgs).to(device)


def prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way):
    support_emb = F.normalize(support_emb, dim=1)
    query_emb = F.normalize(query_emb, dim=1)
    prototypes = torch.stack([support_emb[support_labels == c].mean(0) for c in range(n_way)])
    dists = torch.cdist(query_emb, prototypes)  # (Q, n_way)
    logits = -dists.pow(2)
    loss = F.cross_entropy(logits, query_labels)
    acc = (logits.argmax(1) == query_labels).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate_dogfacenet_val(model, transform, device, batch_size=32):
    split = load_dogfacenet_split(group="val")
    q_feats = extract_features(split.query_paths, model, transform, device, batch_size)
    g_feats = extract_features(split.gallery_paths, model, transform, device, batch_size)
    distmat = compute_distmat(q_feats, g_feats)
    return evaluate_reid(
        distmat, np.asarray(split.query_pids), np.asarray(split.gallery_pids),
        np.asarray(split.query_camids), np.asarray(split.gallery_camids), max_rank=10,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--n-way", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--q-max", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--train-data", choices=["combined", "dogfacenet"], default="combined",
                     help="combined=MPDD+DogFaceNet(기본), dogfacenet=DogFaceNet만 (도메인 불균형 대조실험용)")
    args = ap.parse_args()
    include_mpdd = args.train_data == "combined"

    device = get_device()
    print(f"[info] device={device}", flush=True)

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_ds, num_classes, label_map = build_combined_train_set(train_transform, include_mpdd=include_mpdd)
    class_pools = group_by_label(train_ds.items)
    sampler = EpisodeSampler(class_pools, n_way=args.n_way, k_max=args.k_max, q_max=args.q_max, seed=0)
    print(f"[info] 학습 풀: {num_classes}개체 (2장 미만 제외 후 에피소드 가능 {len(sampler.pools)}개체), "
          f"실제 n_way={sampler.n_way}, train_data={args.train_data}")

    model = ProtoEmbedder(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = "E2_resnet50_protonet.pt" if include_mpdd else "E2_resnet50_protonet_dfnonly.pt"
    best_path = CKPT_DIR / ckpt_name
    best_rank1 = -1.0
    log_rows = []
    running_loss, running_acc, running_n = 0.0, 0.0, 0

    t_start = time.time()
    for ep in range(1, args.episodes + 1):
        model.train()
        support_items, query_items = sampler.sample()
        s_paths, s_labels = zip(*support_items)
        q_paths, q_labels = zip(*query_items)

        s_imgs = load_batch(s_paths, train_transform, device)
        q_imgs = load_batch(q_paths, train_transform, device)
        s_labels_t = torch.tensor(s_labels, device=device)
        q_labels_t = torch.tensor(q_labels, device=device)

        all_emb = model(torch.cat([s_imgs, q_imgs], dim=0))
        s_emb, q_emb = all_emb[:len(s_paths)], all_emb[len(s_paths):]

        loss, acc = prototypical_loss(s_emb, s_labels_t, q_emb, q_labels_t, sampler.n_way)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_acc += acc
        running_n += 1

        if ep % args.log_every == 0:
            print(f"[episode {ep}/{args.episodes}] loss={running_loss/running_n:.3f} "
                  f"acc={running_acc/running_n:.3f} ({time.time()-t_start:.0f}s 경과)")
            running_loss, running_acc, running_n = 0.0, 0.0, 0

        if ep % args.eval_every == 0 or ep == args.episodes:
            metrics = evaluate_dogfacenet_val(model, eval_transform, device)
            print(f"  [val @ episode {ep}] DogFaceNet {metrics}", flush=True)
            log_rows.append({"episode": ep, "dfn_val_rank1": metrics.rank1, "dfn_val_map": metrics.mAP,
                              "seconds": round(time.time() - t_start, 1)})
            if metrics.rank1 > best_rank1:
                best_rank1 = metrics.rank1
                torch.save({"model": model.state_dict(), "episode": ep, "dfn_val_rank1": best_rank1},
                           best_path)
                print(f"  [저장] 새 최고 DogFaceNet val Rank-1={best_rank1:.4f} -> {best_path}", flush=True)

    log_csv = LOG_CSV if include_mpdd else Path("metadata/train_log_E2_dfnonly.csv")
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[완료] 로그: {log_csv}, 최고 체크포인트: {best_path} (DogFaceNet val Rank-1={best_rank1:.4f})", flush=True)


if __name__ == "__main__":
    main()
