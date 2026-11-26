import argparse
import glob
import os
import random

import torch

from eval import iou
from src.detect import Detector
from src.embed import Embedder
from src.video import read_frames


def link(frames, detector, prompt, stride, box_thresh, min_iou=0.3):
    # link boxes by iou
    out, prev, next_id = [], [], 0
    for idx in range(0, len(frames), stride):
        boxes, _ = detector(frames[idx], prompt, box_thresh=box_thresh)
        # keep tall boxes
        boxes = [b for b in boxes if b[3] - b[1] > b[2] - b[0]]
        cur, used = [], set()
        for b in boxes:
            best, score = None, min_iou
            for t, p in prev:
                o = iou(b, p)
                if t not in used and o >= score:
                    best, score = t, o
            if best is None:
                best, next_id = next_id, next_id + 1
            used.add(best)
            cur.append((best, b))
            out.append((idx, best, b))
        prev = cur
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="basketball-51 folder with class subfolders")
    parser.add_argument("--out", default="outputs/b51")
    parser.add_argument("--train-clips", type=int, default=300)
    parser.add_argument("--val-clips", type=int, default=100)
    parser.add_argument("--val-games", type=int, default=8)
    parser.add_argument("--prompt", default="basketball player")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--box-thresh", type=float, default=0.25)
    parser.add_argument("--min-crops", type=int, default=4)
    args = parser.parse_args()

    random.seed(0)
    clips = sorted(glob.glob(os.path.join(args.root, "*", "*.mp4")))
    game = lambda p: os.path.basename(p).split("_")[1]
    # hold out games
    games = sorted({game(p) for p in clips})
    random.shuffle(games)
    val_games = set(games[:args.val_games])
    split = {
        "train": [p for p in clips if game(p) not in val_games],
        "val": [p for p in clips if game(p) in val_games],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(device)
    embedder = Embedder(device)
    os.makedirs(args.out, exist_ok=True)
    for name, count in (("train", args.train_clips), ("val", args.val_clips)):
        paths = random.sample(split[name], count)
        feats, tracks, frames_idx, clip_ids = [], [], [], []
        for c, path in enumerate(paths):
            frames, _ = read_frames(path)
            dets = link(frames, detector, args.prompt, args.stride, args.box_thresh)
            sizes = {}
            for _, t, _ in dets:
                sizes[t] = sizes.get(t, 0) + 1
            dets = [d for d in dets if sizes[d[1]] >= args.min_crops]
            for idx in sorted({d[0] for d in dets}):
                here = [d for d in dets if d[0] == idx]
                feats.append(embedder.features(frames[idx], [d[2] for d in here]).half().cpu())
                tracks += [d[1] for d in here]
                frames_idx += [idx] * len(here)
                clip_ids += [c] * len(here)
            print(f"{name} {c + 1}/{count} {os.path.basename(path)} {len(dets)} crops")
        torch.save({
            "feats": torch.cat(feats),
            "track": torch.tensor(tracks),
            "frame": torch.tensor(frames_idx),
            "clip": torch.tensor(clip_ids),
            "paths": paths,
        }, os.path.join(args.out, f"{name}.pt"))


if __name__ == "__main__":
    main()
