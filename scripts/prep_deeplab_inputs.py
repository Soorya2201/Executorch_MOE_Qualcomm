"""Serialize VOC2012 val images as raw float32 inputs for qnn_executor_runner.

Preprocessing mirrors examples/qualcomm/scripts/deeplab_v3.py:get_dataset().
Writes input_<i>.raw, target_<i>.npy and input_list.txt into the bundle directory.
"""
import argparse
import os

import numpy as np
from torchvision import datasets, transforms

INPUT_SIZE = (224, 224)

parser = argparse.ArgumentParser()
parser.add_argument("--voc_root", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--num", type=int, default=20)
args = parser.parse_args()

preprocess = transforms.Compose(
    [
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
dataset = datasets.VOCSegmentation(
    root=args.voc_root, year="2012", image_set="val", transform=preprocess, download=False
)

inputs_dir = os.path.join(args.out_dir, "inputs")
os.makedirs(inputs_dir, exist_ok=True)
lines = []
# Deterministic, evenly spaced picks across the val split.
step = max(1, len(dataset) // args.num)
for i in range(args.num):
    image, target = dataset[i * step]
    image.unsqueeze(0).numpy().astype(np.float32).tofile(os.path.join(inputs_dir, f"input_{i}.raw"))
    np.save(os.path.join(inputs_dir, f"target_{i}.npy"), np.array(target.resize(INPUT_SIZE)))
    # Relative, space-free paths resolved from the runner's working directory.
    lines.append(f"inputs/input_{i}.raw")

with open(os.path.join(args.out_dir, "input_list.txt"), "w", newline="\n") as f:
    f.write("\n".join(lines))
print(f"wrote {len(lines)} inputs to {inputs_dir}")
