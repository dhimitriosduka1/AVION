import os
import json
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _choose_m(n_pairs_target: int) -> int:
    return int(np.ceil((1 + np.sqrt(1 + 8 * n_pairs_target)) / 2))


def _l2_normalize(X):
    return X


def _gaussian_kernel1d(sigma_bins: float, radius: int | None = None) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(np.ceil(3 * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma_bins**2))
    k /= k.sum()
    return k


def _smooth_density(hist: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    k = _gaussian_kernel1d(sigma_bins)
    return np.convolve(hist, k, mode="same")


def sample_cosine_subset(X, n_pairs_target=500_000, seed=0) -> np.ndarray:
    """Pick m rows uniformly, compute all pairs within (k=1 upper triangle)."""
    rng = np.random.default_rng(seed)
    N, _ = X.shape
    m = min(_choose_m(n_pairs_target), N)
    idx = rng.choice(N, size=m, replace=False)
    Xsub = np.asarray(X[idx], dtype=np.float32)
    Xsub = _l2_normalize(Xsub)
    G = Xsub @ Xsub.T
    iu = np.triu_indices(m, k=1)
    sims = G[iu].astype(np.float32, copy=False)
    return sims


def sample_cosine_uniform_pairs(X, n_pairs=500_000, batch=50_000, seed=0) -> np.ndarray:
    """i.i.d. uniform pairs with j != i; streams from memmap in batches."""
    rng = np.random.default_rng(seed)
    N, _ = X.shape
    out = []
    remain = n_pairs
    while remain > 0:
        k = min(batch, remain)
        i = rng.integers(0, N, size=k)
        j = rng.integers(0, N - 1, size=k)
        j += j >= i

        Xi = _l2_normalize(np.asarray(X[i], dtype=np.float32))
        Xj = _l2_normalize(np.asarray(X[j], dtype=np.float32))
        sims = np.einsum("ij,ij->i", Xi, Xj, dtype=np.float32)
        out.append(sims.astype(np.float32, copy=False))
        remain -= k
    return np.concatenate(out, dtype=np.float32)


def plot_cosine_density_from_memmap(
    path: str,
    dtype: str = "float32",
    method: str = "subset",
    n_pairs: int = 500_000,
    bins: int = 200,
    smooth_sigma_bins: float = 1.2,
    seed: int = 42,
):
    """
    Opens the memmap, samples cosine similarities (excluding self-pairs),
    builds a density, and shows a smoothed curve.
    """
    shape = json.load(open(os.path.join(path, "shape.json"), "r"))["shape"]
    X = np.memmap(
        f"{path}/embeddings.memmap", dtype=dtype, mode="r", shape=tuple(shape)
    )

    if method == "subset":
        sims = sample_cosine_subset(X, n_pairs_target=n_pairs, seed=seed)
    elif method == "pairs":
        sims = sample_cosine_uniform_pairs(X, n_pairs=n_pairs, seed=seed)
    else:
        raise ValueError("method must be 'subset' or 'pairs'")

    mean = float(np.mean(sims))
    std = float(np.std(sims))

    hist, edges = np.histogram(sims, bins=bins, range=(-1.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_smooth = _smooth_density(hist, smooth_sigma_bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, hist, label="Density")
    ax.plot(centers, hist_smooth, label=f"Smoothed")

    ax.axvline(mean, linestyle="--", linewidth=1.5, label=f"(mean = {mean:.4f})")
    ax.axvspan(mean - std, mean + std, alpha=0.15, label=f"(std = {std:.4f})")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity Density")
    ax.legend()
    fig.tight_layout()

    return fig


def main(args):

    wandb.init(
        project="Thesis",
        name=f"Similarity Distribution - {args.source_run_name}",
        config={**args.__dict__},
        group=f"Text Embeddings",
    )

    print("Computing similarity distribution")
    fig = plot_cosine_density_from_memmap(
        path=args.embeddings_path,
        dtype=args.dtype,
        method=args.method,
        n_pairs=args.n_pairs,
        bins=args.bins,
        smooth_sigma_bins=args.smooth_sigma_bins,
        seed=args.seed,
    )

    wandb.log(
        {
            "similarity_distribution": wandb.Image(fig),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-name", type=str, required=True)
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--n-pairs", type=int, required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--smooth-sigma-bins", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    main(args)
