import argparse
import time

import numpy as np
import torch
import torch.nn as nn

voronoi = "${DEV_ROOT}/phace-atomic/voronoi"
run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"
N_CORESET = 2 ** 24

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="p20d8")
parser.add_argument("--d", type=int, default=8)
parser.add_argument("--chunk", type=int, default=262_144)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    f"{run_dir}/projector/projector_d{args.d}_{args.tag}.pt",
    map_location=device, weights_only=False)

layers, width = [], 512
for hidden_width in checkpoint["hidden"]:
    layers += [nn.Linear(width, hidden_width), nn.ReLU()]
    width = hidden_width
layers.append(nn.Linear(width, checkpoint["d"]))
model = nn.Sequential(*layers).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
feature_scale = torch.as_tensor(checkpoint["feature_scale"], device=device)
projection_mean = np.asarray(checkpoint["projection_mean"], dtype=np.float32)
projection_scale = np.asarray(checkpoint["projection_scale"], dtype=np.float32)

features = np.memmap(f"{voronoi}/out/tree/center_feats_scaled.f32", dtype=np.float32,
                     mode="r", shape=(N_CORESET, 512))
projection = np.empty((N_CORESET, checkpoint["d"]), dtype=np.float32)

started = time.time()
with torch.no_grad():
    for start in range(0, N_CORESET, args.chunk):
        end = min(start + args.chunk, N_CORESET)
        block = torch.as_tensor(np.ascontiguousarray(features[start:end]), device=device)
        block = (block - feature_mean) / feature_scale
        projection[start:end] = model(block).cpu().numpy()
        if (start // args.chunk) % 16 == 0:
            print(f"  {end:,}/{N_CORESET:,}  {time.time()-started:.0f}s", flush=True)

projection = projection * projection_scale + projection_mean

out = f"{run_dir}/projector/projection_2p24_d{args.d}_{args.tag}.npy"
np.save(out, projection)
print(f"\nwrote {out}  shape {projection.shape}  {time.time()-started:.0f}s")

fitted = np.load(f"{run_dir}/results/embeddings/embedding_l20_d{args.d}_{args.tag}.npy")
residual = np.abs(projection[:len(fitted)] - fitted)
print(f"agreement with the fitted 2^20 embedding: "
      f"median |delta| {np.median(residual):.4f}, "
      f"95th percentile {np.percentile(residual, 95):.4f}, "
      f"embedding scale {fitted.std():.4f}")
