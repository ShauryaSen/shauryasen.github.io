---
title: "Variational Autoencoders, with a Latent Space You Can Click"
date: 2026-10-14
draft: false
math: true
tags:
  - deep-learning
  - generative-models
  - vae
summary: "From autoencoders to the ELBO, with a 2D-latent VAE I trained on MNIST running live in the page."
---

**Click or drag anywhere in the plot.** Each dot is one handwritten digit, compressed to a point in a 2D latent space.

{{< vae-latent >}}

A normal autoencoder can't do this: click in the gaps of its latent space and you get static. The rest of this post explains why a VAE's latent space is smooth enough to wander around in.

## Data manifolds

The manifold hypothesis says that real data, the inputs that actually make sense to us, sit on a low-dimensional manifold inside the high-dimensional space they're written in. A $28\times 28$ image lives in $\mathbb{R}^{784}$, but almost every point in $\mathbb{R}^{784}$ is static. The handwritten digits are a thin, curved sheet inside it.

Locally, that sheet can be "unrolled": near any point, a few directions are enough to move along it. So sampling uniformly from the ambient space almost never lands on a digit.

The way around this is to sample from a simple distribution we *can* sample from, like a Gaussian, and learn a map from it onto the data manifold. VAEs learn a smooth probabilistic version of that map. GANs learn a deterministic one through an adversarial game.

## Dimensionality reduction

The core of a VAE is compressing $x\in\mathbb{R}^D$ into a latent $z\in\mathbb{R}^d$ with $d\ll D$. A good compression should:

1. keep the structure that matters, so similar inputs get nearby codes,
2. keep enough information to reconstruct $x$ from $z$, and
3. for a VAE specifically, make the latent space easy to sample from.

### PCA

PCA is the linear version. It assumes the data lies near a linear subspace and picks the projection $W$ that keeps the most variance. That works for feature extraction, but real manifolds curve, and a linear projection flattens the curve.

### Autoencoders

An autoencoder does the nonlinear version with two networks:

- an **encoder** $e_\phi(x)$ that maps $x\in\mathbb{R}^D$ to $z\in\mathbb{R}^d$, and
- a **decoder** $d_\theta(z)$ that maps back to a reconstruction $x'\approx x$,

trained to minimize reconstruction error:

$$
\min_{\phi,\theta}\ \sum_{x\in\mathcal{D}} \Vert x-x'\Vert^2, \qquad x'=d_\theta(e_\phi(x)).
$$

### Why you can't sample from an autoencoder

To generate new data, we'd like to draw $z\sim p(z)$ and decode it. But nothing in the autoencoder's loss says what $p(z)$ is. Training only cares about the codes of the training points, so the latent space ends up:

1. **sparse**: codes cluster wherever is convenient, with big gaps between them, and
2. **unstructured**: there's no known shape to sample from, and points in the gaps decode to garbage.

A VAE fixes both by making the encoder output a *distribution* and pinning those distributions to a known prior.

## KL divergence

The tool for "pinning one distribution to another" is the KL divergence:

$$
D_{\mathrm{KL}}(p\,\Vert\,q)=\mathbb{E}_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right].
$$

One reading is the expected *extra surprise* from modeling data from $p$ with $q$. It's always $\ge 0$, it's $0$ only when $p=q$, and it isn't symmetric. The direction matters:

- **Forward KL, $D_{\mathrm{KL}}(p\Vert q)$, is mass-covering.** The expectation is over $p$, so wherever $p$ has mass and $q\approx 0$, $\log\frac{p}{q}$ explodes. $q$ is forced to cover all of $p$, even if that means spreading thin.
- **Reverse KL, $D_{\mathrm{KL}}(q\Vert p)$, is mode-seeking.** Now the expectation is over $q$, so the penalty is for $q$ putting mass where $p$ has none. $q$ stays tucked under $p$, often inside a single mode.

The VAE objective below uses $D_{\mathrm{KL}}(q_\phi(z\mid x)\,\Vert\,p(z))$.

## Making the autoencoder probabilistic

### Probabilistic encoder

Instead of one code per input, the encoder outputs a Gaussian over codes:

$$
q_\phi(z\mid x)=\mathcal{N}\big(\mu_\phi(x),\ \mathrm{diag}(\sigma^2_\phi(x))\big).
$$

In practice the network predicts $\mu$ and $\log\sigma^2$, since the log keeps the variance positive without a constraint.

### Probabilistic decoder

The decoder defines $p_\theta(x\mid z)$. With a Gaussian decoder of fixed variance, maximizing $\log p_\theta(x\mid z)$ is the same as minimizing squared error:

$$
\max_{\phi,\theta}\ \mathbb{E}_{z\sim q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]
\quad\Longleftrightarrow\quad
\min_{\phi,\theta}\ \mathbb{E}_{z\sim q_\phi(z\mid x)}\big[\Vert x-d_\theta(z)\Vert^2\big].
$$

For MNIST I used a Bernoulli decoder instead, which treats each pixel as a coin flip. That makes the reconstruction term binary cross-entropy.

### Keeping every $q_\phi(z\mid x)$ near one prior

Each input now gets its own little Gaussian. If those Gaussians drift anywhere they like, we're back to a sparse, unstructured latent space. So the VAE also penalizes each one for straying from a shared prior, $p(z)=\mathcal{N}(0,I)$:

$$
D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big).
$$

That's why the cloud in the demo sits roughly inside the unit-Gaussian region and has no gaping holes.

## The evidence lower bound

What we'd actually like to maximize is the likelihood of the data, $\log p_\theta(x)=\log\int p_\theta(x\mid z)\,p(z)\,dz$. That integral is over every possible $z$, so it's intractable. Instead we maximize a lower bound on it, the **ELBO**. The idea is to push a floor up as high as it will go; the quantity above it has to rise with it.

> [!derivation]- Deriving the ELBO
>
> Start from the log-evidence and bring in $q_\phi(z\mid x)$, which integrates to one:
>
> $$
> \begin{aligned}
> \log p_\theta(x)
>   &= \int q_\phi(z\mid x)\,\log p_\theta(x)\,dz
>   && \textstyle\int q_\phi = 1 \\[6pt]
>   &= \mathbb{E}_{q_\phi}\left[\log\frac{p_\theta(x,z)}{p_\theta(z\mid x)}\right]
>   && \text{Bayes: } p(x)=\tfrac{p(x,z)}{p(z\mid x)} \\[6pt]
>   &= \mathbb{E}_{q_\phi}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\cdot\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\right]
>   && \text{multiply and divide by } q_\phi \\[6pt]
>   &= \underbrace{\mathbb{E}_{q_\phi}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]}_{\text{ELBO}}
>      + \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\big)}_{\ge\,0}
>   && \text{split the log}
> \end{aligned}
> $$
>
> The KL term is non-negative, so $\log p_\theta(x)\ge\text{ELBO}$. The gap is exactly how far the encoder is from the true posterior. Raising the ELBO either raises the likelihood or tightens the gap, and both are things we want.

Expanding $p_\theta(x,z)=p_\theta(x\mid z)\,p(z)$ splits the ELBO into the two terms you actually implement:

> [!result] The VAE objective
>
> $$
> \text{ELBO}(x)=\underbrace{\mathbb{E}_{z\sim q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}-\underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{stay near the prior}}
> $$
>
> The first term asks the decoder to rebuild $x$ from a sampled code. The second keeps every code distribution close to $\mathcal{N}(0,I)$. They pull in opposite directions: perfect reconstructions want tight, well-separated $q$'s, and the prior wants them all to be the same blob.

For a diagonal Gaussian encoder and a standard normal prior, the KL term has a closed form, so no sampling is needed for it:

> [!derivation]- Closed-form KL for diagonal Gaussians
>
> Per latent dimension, with $q=\mathcal{N}(\mu,\sigma^2)$ and $p=\mathcal{N}(0,1)$:
>
> $$
> \begin{aligned}
> D_{\mathrm{KL}}(q\Vert p)
>   &= \mathbb{E}_q\big[\log q(z)-\log p(z)\big] \\[6pt]
>   &= \mathbb{E}_q\left[-\tfrac12\log\sigma^2-\frac{(z-\mu)^2}{2\sigma^2}+\frac{z^2}{2}\right]
>   && \text{the } \log\sqrt{2\pi} \text{ terms cancel} \\[6pt]
>   &= -\tfrac12\log\sigma^2-\tfrac12+\tfrac12(\sigma^2+\mu^2)
>   && \mathbb{E}_q[(z-\mu)^2]=\sigma^2,\ \mathbb{E}_q[z^2]=\sigma^2+\mu^2 \\[6pt]
>   &= -\tfrac12\big(1+\log\sigma^2-\mu^2-\sigma^2\big)
> \end{aligned}
> $$
>
> Sum over the $d$ dimensions. This is the `kl` line in the code below.

## The reparameterization trick

The reconstruction term is an expectation over $z\sim q_\phi(z\mid x)$, and we need its gradient with respect to $\phi$. You can't backpropagate through "draw a random sample." The trick is to move the randomness outside the network:

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).
$$

Now $z$ is a deterministic, differentiable function of $\mu$ and $\sigma$, plus noise that doesn't depend on $\phi$. Gradients flow straight through. This is also much lower variance than the score-function (REINFORCE-style) estimator you'd otherwise need.

## The implementation

This is the model and loss behind the demo at the top. The full script, including the export to the browser, is [`train_vae.py`](/demos/vae/train_vae.py).

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.to_mu = nn.Linear(256, 2)
        self.to_logvar = nn.Linear(256, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784),  # logits
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # reparameterization
        return self.decoder(z), mu, logvar


def neg_elbo(logits, x, mu, logvar):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")  # -log p(x|z)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())           # KL(q(z|x) || N(0,I))
    return recon + kl
```

Trained with Adam (learning rate $10^{-3}$, batch size 128) for 40 epochs, which took a couple of minutes on a laptop. It ends at a −ELBO of about 138 nats per image, roughly 132 of reconstruction and 7 of KL. The decoder is about 218k parameters. Stored as float16 that's 426 KB, which the page loads and runs in plain JavaScript.

## What the demo shows, and what it doesn't

- **The gaps decode to real digits** because every $q_\phi(z\mid x)$ is pulled toward the same $\mathcal{N}(0,I)$ prior. That leaves no holes for the decoder to ignore, even out near the corners.
- **Smooth interpolation** is the KL term at work. Drag from one cluster to a neighboring one and the digit morphs instead of jumping. Neighboring codes have to overlap, so the decoder learns to make the space between them meaningful.
- **Blurry digits** are the classic VAE complaint. The decoder is rewarded for the *average* reconstruction over a neighborhood of $z$, and the average of several plausible 4s is a smudged 4. Two latent dimensions make it worse; with 16 or 32 the samples get much sharper, but you can't plot the space anymore.
- **The decoded grid** is the decoder evaluated on an even $20\times20$ lattice of $z$. With only two dimensions, the model has to lay all ten digits out on one sheet.
- **Overlapping clusters** (4/9 and 3/5/8 in particular) are honest. Those digits really are close in pixel space, and two dimensions don't leave room to separate them.

---

*Credit: these notes follow Cornell's [CS 4782: Deep Learning](https://www.cs.cornell.edu/courses/cs4782/2026sp/), which I took in Spring 2026, especially the course's [U-Nets and VAEs notes](https://courses.cis.cornell.edu/courses/cs4782/2026sp/notes/). The interactive demo here is my own reimplementation of the idea behind the course's [VAE latent-space demo](https://www.cs.cornell.edu/courses/cs4782/2026sp/demos/vae/vae_viewer.html), which is worth a look too. Mistakes are mine.*
