import argparse
import os
import random

import torch
import torch.nn.functional as F

from src.embed import make_head


def groups(data):
    # crops per track
    out = {}
    for i, (c, t) in enumerate(zip(data["clip"].tolist(), data["track"].tolist())):
        out.setdefault(c, {}).setdefault(t, []).append(i)
    clips = [[t for t in g.values() if len(t) >= 2] for g in out.values()]
    return [c for c in clips if len(c) >= 2]


def embed(head, x):
    x = x.float()
    return F.normalize(head(x) if head is not None else x, dim=-1)


@torch.no_grad()
def evaluate(head, data, clips):
    # first vs last crop
    hits, total, pos, neg = 0, 0, [], []
    for tracks in clips:
        ref = embed(head, data["feats"][[t[0] for t in tracks]])
        query = embed(head, data["feats"][[t[-1] for t in tracks]])
        sims = query @ ref.T
        hits += int((sims.argmax(1) == torch.arange(len(tracks))).sum())
        total += len(tracks)
        pos += sims.diag().tolist()
        neg += sims[~torch.eye(len(tracks), dtype=torch.bool)].tolist()
    return hits / total, sum(pos) / len(pos), sum(neg) / len(neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/b51")
    parser.add_argument("--out", default="outputs/b51/head.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clips-per-step", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.07)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    train = torch.load(os.path.join(args.data, "train.pt"))
    val = torch.load(os.path.join(args.data, "val.pt"))
    train_clips, val_clips = groups(train), groups(val)
    print(f"train {len(train_clips)} clips {len(train['feats'])} crops")
    print(f"val {len(val_clips)} clips {len(val['feats'])} crops")

    head = make_head()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    acc, p, n = evaluate(None, val, val_clips)
    print(f"plain clip val acc {acc:.3f} pos {p:.3f} neg {n:.3f}")
    best = 0
    for epoch in range(args.epochs):
        random.shuffle(train_clips)
        head.train()
        total = 0
        for s in range(0, len(train_clips), args.clips_per_step):
            loss = 0
            for tracks in train_clips[s:s + args.clips_per_step]:
                # other players are negatives
                pairs = [random.sample(t, 2) for t in tracks]
                a = embed(head, train["feats"][[x[0] for x in pairs]])
                b = embed(head, train["feats"][[x[1] for x in pairs]])
                logits = a @ b.T / args.temp
                target = torch.arange(len(tracks))
                loss = loss + (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        head.eval()
        acc, p, n = evaluate(head, val, val_clips)
        print(f"epoch {epoch + 1} loss {total:.2f} val acc {acc:.3f} pos {p:.3f} neg {n:.3f}")
        if acc > best:
            best = acc
            torch.save(head.state_dict(), args.out)
    print(f"best val acc {best:.3f} saved {args.out}")


if __name__ == "__main__":
    main()
