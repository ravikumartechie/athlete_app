"""STEP 7: make the 4 report figures into a 'figures' folder."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import engine as E

os.makedirs("figures", exist_ok=True)
df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))

# ---------- Figure 1: PCA scatter (7 features squeezed into 2) ----------
pca = PCA(n_components=2, random_state=0)
xy = pca.fit_transform(df[E.PCT].values)
groups = sorted(df["sport_group"].unique())
fig, ax = plt.subplots(figsize=(8, 6))
for g in groups:
    m = (df["sport_group"] == g).values
    ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, label=g)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
ax.set_title("PCA of normalized athlete profiles (coloured by sport group)")
ax.legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig("figures/pca_scatter.png", dpi=200); plt.close()

# ---------- Figure 2: dendrograms of the 20 sports, one per metric ----------
sports = sorted(df["sport"].unique())
centres = np.array([df[df.sport == s][E.PCT].mean().values for s in sports])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, metric) in zip(axes, [("Euclidean", "euclidean"), ("Cosine", "cosine")]):
    Z = linkage(centres, method="average", metric=metric)
    dendrogram(Z, labels=sports, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"Sport similarity tree: {name}")
plt.tight_layout(); plt.savefig("figures/dendrogram.png", dpi=200); plt.close()

# ---------- Figure 3: clustering scores ----------
cs = E.clustering_scores(df)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, col in zip(axes, ["Silhouette", "Adjusted Rand vs true sports"]):
    ax.bar(cs["Metric"], cs[col], color=["#4C72B0", "#DD8452"])
    ax.set_title(col)
    for i, v in enumerate(cs[col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig("figures/metric_comparison.png", dpi=200); plt.close()

# ---------- Figure 4: transfer-validation accuracy ----------
tr = E.transfer_results(df)
x = np.arange(len(tr)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(x - w/2, tr["Euclid top-1"], w, label="Euclidean", color="#4C72B0")
ax.bar(x + w/2, tr["Cosine top-1"], w, label="Cosine", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(tr["Pair"], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Top-1 accuracy"); ax.set_ylim(0, 1)
ax.set_title(f"Transfer validation (overall: Euclidean {tr['Euclid top-1'].mean():.0%}, "
             f"Cosine {tr['Cosine top-1'].mean():.0%})")
ax.legend(); plt.tight_layout(); plt.savefig("figures/transfer_accuracy.png", dpi=200); plt.close()

print("Saved 4 figures in the 'figures' folder:")
for f in sorted(os.listdir("figures")):
    print(" -", f)