import os
import json
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt


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


def _choose_mx_my(n_pairs_target: int, N: int, M: int) -> tuple[int, int]:
    """
    Choose sample sizes m_x and m_y so that m_x * m_y ≈ n_pairs_target,
    while respecting m_x <= N and m_y <= M and roughly preserving the N:M ratio.
    """
    T = int(max(1, n_pairs_target))
    # If target exceeds all possible cross-pairs, take all.
    if T >= N * M:
        return N, M

    # Aim to preserve sampling proportion m_x/N ≈ m_y/M.
    mx = int(np.floor(np.sqrt(T * N / max(1, M))))
    mx = max(1, min(N, mx))
    my = int(np.ceil(T / mx))
    my = max(1, min(M, my))

    # If product overshoots too much (rare), trim slightly.
    while mx * my > T and (mx > 1 or my > 1):
        # Reduce the side sampled "too heavily" relative to its population.
        if (mx / max(1, N)) >= (my / max(1, M)) and mx > 1:
            mx -= 1
        elif my > 1:
            my -= 1
        else:
            break

    # If we undershot and still can increase, bump the smaller relative side.
    while mx * my < min(T, N * M):
        if (mx / max(1, N)) <= (my / max(1, M)) and mx < N:
            mx += 1
        elif my < M:
            my += 1
        else:
            break

    return mx, my


def sample_cross_cosine_between(
    X: np.ndarray,
    Y: np.ndarray,
    n_pairs_target: int = 500_000,
    seed: int = 0,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Uniformly sample rows from X and Y, L2-normalize, and return all cross
    cosine similarities between the sampled rows.

    Returns
    -------
    sims : (m_x * m_y,) float32
        Flattened (row-major) similarities for every (i in Xsub, j in Ysub).
        Position p maps to (i, j) via: i = p // m_y, j = p % m_y.

    If return_indices=True, also returns:
        idx_x : (m_x,) sampled row indices from X
        idx_y : (m_y,) sampled row indices from Y
        shape_xy : (m_x, m_y) to reshape sims into a matrix if desired
    """
    rng = np.random.default_rng(seed)

    N, dx = X.shape
    M, dy = Y.shape
    if dx != dy:
        raise ValueError(f"Dim mismatch: X has d={dx}, Y has d={dy}")

    m_x, m_y = _choose_mx_my(n_pairs_target, N, M)

    idx_x = rng.choice(N, size=m_x, replace=False)
    idx_y = rng.choice(M, size=m_y, replace=False)

    Xsub = _l2_normalize(np.asarray(X[idx_x], dtype=np.float32))
    Ysub = _l2_normalize(np.asarray(Y[idx_y], dtype=np.float32))

    # Cross cosine similarities = dot product of normalized vectors
    G = Xsub @ Ysub.T  # (m_x, m_y)
    sims = G.astype(np.float32, copy=False).ravel()  # (m_x*m_y,)

    if return_indices:
        return sims, idx_x, idx_y, (m_x, m_y)
    return sims


def plot_cosine_density_from_memmap(
    embeddings_path: str,
    target_embeddings_path: str,
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
    shape = json.load(open(os.path.join(embeddings_path, "shape.json"), "r"))["shape"]
    embeddings = np.memmap(
        f"{embeddings_path}/embeddings.memmap",
        dtype=dtype,
        mode="r",
        shape=tuple(shape),
    )

    shape = json.load(open(os.path.join(target_embeddings_path, "shape.json"), "r"))[
        "shape"
    ]
    target_embeddings = np.memmap(
        f"{target_embeddings_path}/embeddings.memmap",
        dtype=dtype,
        mode="r",
        shape=tuple(shape),
    )

    if method == "subset":
        sims = sample_cross_cosine_between(
            embeddings, target_embeddings, n_pairs_target=n_pairs, seed=seed
        )
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
        name=f"Similarity Distribution - {args.source_run_name} vs {args.target_run_name}",
        config={**args.__dict__},
        group=f"Text Embeddings",
    )

    print("Computing similarity distribution")
    fig = plot_cosine_density_from_memmap(
        embeddings_path=args.embeddings_path,
        target_embeddings_path=args.target_embeddings_path,
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
    parser.add_argument("--target-run-name", type=str, required=True)
    parser.add_argument("--target-embeddings-path", type=str, required=True)
    parser.add_argument("--method", type=str, default="subset")
    parser.add_argument("--n-pairs", type=int, default=500_000)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--smooth-sigma-bins", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    main(args)
