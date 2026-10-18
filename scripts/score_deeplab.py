"""Score qnn_executor_runner DeepLab outputs against VOC ground truth (PA / mIoU).

Output layout follows examples/qualcomm/scripts/deeplab_v3.py: output_<i>_0.raw is
float32 logits [21, 224, 224]; output_<i>_1.raw is the auxiliary head and is ignored.
"""
import argparse
import os

import numpy as np

NUM_CLASSES = 21

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", required=True)
parser.add_argument("--num", type=int, default=20)
args = parser.parse_args()

hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for i in range(args.num):
    logits = np.fromfile(os.path.join(args.bundle, "outputs", f"output_{i}_0.raw"), dtype=np.float32)
    pred = logits.reshape(NUM_CLASSES, 224, 224).argmax(0)
    target = np.load(os.path.join(args.bundle, "inputs", f"target_{i}.npy"))
    valid = target != 255  # VOC "void" boundary pixels
    hist += np.bincount(
        NUM_CLASSES * target[valid].astype(np.int64) + pred[valid], minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)

pa = np.diag(hist).sum() / hist.sum()
with np.errstate(divide="ignore", invalid="ignore"):
    iou = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist))
present = hist.sum(1) > 0  # classes that appear in these images
print(f"images scored     : {args.num}")
print(f"pixel accuracy    : {pa * 100:.1f}%")
print(f"mIoU (present cls): {np.nanmean(iou[present]) * 100:.1f}%  over {present.sum()} classes")
