import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.distance import pdist

voronoi = "${DEV_ROOT}/phace-atomic/voronoi"
run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="p20d8")
parser.add_argument("--embedding-tag", default=None)
parser.add_argument("--d", type=int, default=8)
parser.add_argument("--level", type=int, default=20)
parser.add_argument("--hidden", type=int, nargs="*", default=[512, 256, 128])
parser.add_argument("--activation", default="relu", choices=["relu", "silu"])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--validation-fraction", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=20)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
n_training_points = 2 ** args.level

features = np.memmap(f"{voronoi}/out/tree/center_feats_scaled.f32", dtype=np.float32,
                     mode="r", shape=(2 ** 24, 512))
embedding_tag = args.embedding_tag or args.tag
projection = np.load(f"{run_dir}/results/embeddings/embedding_l20_d{args.d}_{embedding_tag}.npy")
assert projection.shape == (n_training_points, args.d), projection.shape

inputs = np.ascontiguousarray(features[:n_training_points])
feature_mean, feature_scale = inputs.mean(0), inputs.std(0) + 1e-8
projection_mean, projection_scale = projection.mean(0), projection.std(0) + 1e-8

inputs = torch.as_tensor((inputs - feature_mean) / feature_scale)
targets = torch.as_tensor((projection - projection_mean) / projection_scale, dtype=torch.float32)

generator = torch.Generator().manual_seed(0)
shuffled = torch.randperm(n_training_points, generator=generator)
n_validation = int(args.validation_fraction * n_training_points)
validation_index, training_index = shuffled[:n_validation], shuffled[n_validation:]

activation = {"relu": nn.ReLU, "silu": nn.SiLU}[args.activation]

layers, width = [], inputs.shape[1]
for hidden_width in args.hidden:
    layers += [nn.Linear(width, hidden_width), activation()]
    width = hidden_width
layers.append(nn.Linear(width, args.d))
model = nn.Sequential(*layers).to(device)

print(f"device={device}  train={len(training_index):,}  "
      f"validation={len(validation_index):,}", flush=True)
print(model, flush=True)

optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=5)
loss_function = nn.MSELoss()

validation_inputs = inputs[validation_index].to(device)
validation_targets = targets[validation_index].to(device)

best_loss, best_state, waited = float("inf"), None, 0
started = time.time()

for epoch in range(args.epochs):
    model.train()
    order = training_index[torch.randperm(len(training_index), generator=generator)]
    running = 0.0
    for start in range(0, len(order), args.batch_size):
        batch = order[start:start + args.batch_size]
        prediction = model(inputs[batch].to(device))
        loss = loss_function(prediction, targets[batch].to(device))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running += loss.item() * len(batch)

    model.eval()
    with torch.no_grad():
        validation_loss = loss_function(model(validation_inputs), validation_targets).item()
    schedule.step(validation_loss)

    if validation_loss < best_loss - 1e-6:
        best_loss, waited = validation_loss, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        waited += 1

    if epoch % 10 == 0 or waited >= args.patience:
        print(f"epoch {epoch:4d}  train {running/len(order):.5f}  "
              f"validation {validation_loss:.5f}  best {best_loss:.5f}  "
              f"{time.time()-started:.0f}s", flush=True)

    if waited >= args.patience:
        print(f"early stop at epoch {epoch}", flush=True)
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    predicted = model(validation_inputs).cpu().numpy()

actual = targets[validation_index].numpy()
residual = ((predicted - actual) ** 2).sum(0)
variance = ((actual - actual.mean(0)) ** 2).sum(0)
r_squared = 1.0 - residual / variance

print(f"\nvalidation R^2 per dimension: "
      f"{np.array2string(r_squared, precision=4)}")
print(f"mean R^2: {r_squared.mean():.4f}")

sample = np.sort(np.random.default_rng(0).choice(len(validation_index), 4000, replace=False))

distance_predicted = pdist(predicted[sample].astype(np.float64))
distance_actual = pdist(actual[sample].astype(np.float64))
correlation = float(np.corrcoef(distance_predicted, distance_actual)[0, 1])
print(f"pairwise-distance correlation on held-out points: {correlation:.4f}")

torch.save({"state_dict": best_state, "hidden": args.hidden, "d": args.d,
            "activation": args.activation,
            "feature_mean": feature_mean, "feature_scale": feature_scale,
            "projection_mean": projection_mean, "projection_scale": projection_scale},
           f"{run_dir}/projector/projector_d{args.d}_{args.tag}.pt"
        )

with open(f"{run_dir}/projector/projector_d{args.d}_{args.tag}.json", "w") as fh:
    json.dump({"tag": args.tag, "d": args.d, "hidden": args.hidden,
               "activation": args.activation,
               "epochs_run": epoch + 1, "best_validation_mse": best_loss,
               "r_squared_per_dimension": r_squared.tolist(),
               "mean_r_squared": float(r_squared.mean()),
               "distance_correlation": correlation}, fh, indent=1)
print(f"\nsaved projector_d{args.d}_{args.tag}.pt")
