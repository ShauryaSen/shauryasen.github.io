"""Train a 2D-latent VAE on MNIST and export the decoder for the in-browser demo.

    python train_vae.py --data ./data

Writes decoder.bin (float16 weights) and latents.json (encoder means for test digits).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LATENT_DIM = 2  # 2D so the whole latent space fits on screen


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.to_mu = nn.Linear(256, LATENT_DIM)
        self.to_logvar = nn.Linear(256, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784),  # logits; sigmoid gives pixel intensities
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Reparameterization: z = mu + sigma * eps keeps the sample differentiable in mu, sigma.
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decoder(z), mu, logvar


def neg_elbo(logits, x, mu, logvar):
    # Reconstruction: -log p(x|z) under a Bernoulli decoder.
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    # KL(q(z|x) || N(0, I)) in closed form for diagonal Gaussians.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default=".")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    flat = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    train = datasets.MNIST(args.data, train=True, download=True, transform=flat)
    test = datasets.MNIST(args.data, train=False, download=True, transform=flat)
    loader = DataLoader(train, batch_size=128, shuffle=True)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        totals = np.zeros(3)
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss, recon, kl = neg_elbo(logits, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totals += [loss.item(), recon.item(), kl.item()]
        n = len(train)
        print(f"epoch {epoch + 1:2d}  -ELBO {totals[0] / n:6.1f}  recon {totals[1] / n:6.1f}  KL {totals[2] / n:4.1f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()

    # Decoder weights, layer by layer, as little-endian float16: W (out x in, row-major) then b.
    layers = [m for m in model.decoder if isinstance(m, nn.Linear)]
    blob = b"".join(
        t.detach().numpy().astype("<f2").tobytes() for m in layers for t in (m.weight, m.bias)
    )
    (out / "decoder.bin").write_bytes(blob)

    # Encoder means for 4,000 test digits, for the scatter plot.
    xs = torch.stack([test[i][0] for i in range(4000)])
    labels = [test[i][1] for i in range(4000)]
    with torch.no_grad():
        mu, _ = model.encode(xs)
    json.dump(
        {
            "layers": [[m.in_features, m.out_features] for m in layers],
            "points": [[round(a, 2), round(b, 2), y] for (a, b), y in zip(mu.tolist(), labels)],
        },
        open(out / "latents.json", "w"),
        separators=(",", ":"),
    )
    print(f"wrote {len(blob) / 1024:.0f} KB of decoder weights to {out}")


if __name__ == "__main__":
    main()
