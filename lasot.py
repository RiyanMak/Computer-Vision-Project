import argparse
import csv
import os

import cv2
import numpy as np
from remotezip import RemoteZip

from src.video import write_video

URL = "https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/{}.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    cat = args.seq.split("-")[0]
    os.makedirs(args.out, exist_ok=True)
    with RemoteZip(URL.format(cat)) as z:
        base = next(n for n in z.namelist() if n.endswith(f"{args.seq}/groundtruth.txt"))[:-15]
        read = lambda f: z.read(base + f).decode().split()
        gt = [[int(v) for v in line.split(",")] for line in read("groundtruth.txt")]
        absent = [int(a) or int(b) for a, b in zip(",".join(read("full_occlusion.txt")).split(","),
                                                    ",".join(read("out_of_view.txt")).split(","))]
        prompt = " ".join(read("nlp.txt"))
        frames = []
        for i in range(args.start, args.start + args.length):
            data = np.frombuffer(z.read(f"{base}img/{i + 1:08d}.jpg"), np.uint8)
            frames.append(cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

    write_video(os.path.join(args.out, f"{args.seq}.mp4"), frames, 30)
    with open(os.path.join(args.out, f"{args.seq}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "absent"])
        for i in range(0, args.length, args.every):
            x, y, w, h = gt[args.start + i]
            if absent[args.start + i]:
                writer.writerow([i, "", "", "", "", 1])
            else:
                writer.writerow([i, x, y, x + w, y + h, 0])
    print(f"{args.seq},{prompt}")


if __name__ == "__main__":
    main()
