"""Calculation engine. Works on ANY CSV with the required columns."""
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

FEATURES = ["sprint_30m_s", "vertical_jump_cm", "broad_jump_cm", "vo2max",
            "grip_kg", "agility_s", "flexibility_cm"]
LOWER_IS_BETTER = {"sprint_30m_s", "agility_s"}
REQUIRED = ["sport", "sex"] + FEATURES
PCT = ["pct_" + f for f in FEATURES]


def load_csv(path):
    """Read a dataset, check it has the needed columns, and clean it."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    df["sex"] = df["sex"].astype(str).str.upper().str[0]
    return df


def normalize_percentile(df):
    """Percentile rank (0 to 1) within each sex, pooled across all sports."""
    out = df.copy()
    for feat in FEATURES:
        pct = pd.Series(index=df.index, dtype=float)
        for sex, group in df.groupby("sex"):
            ranks = group[feat].rank(method="average")
            p = (ranks - 1) / (len(group) - 1)
            if feat in LOWER_IS_BETTER:
                p = 1 - p                      # flip: faster time = higher score
            pct.loc[group.index] = p
        out[f"pct_{feat}"] = pct
    return out
def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sport_centroids(df):
    """The 'average athlete' of each sport and sex (mean percentile vector)."""
    return {(sport, sex): group[PCT].mean().values
            for (sport, sex), group in df.groupby(["sport", "sex"])}


def compare_to_sports(vec, centroids, sex):
    """Compare one athlete's percentile vector to every sport, using both metrics."""
    rows = []
    for (sport, s), centre in centroids.items():
        if s != sex:
            continue
        rows.append({"sport": sport,
                     "euclid_dist": euclidean_distance(vec, centre),
                     "cosine_sim": cosine_similarity(vec, centre)})
    t = pd.DataFrame(rows)
    t["euclid_rank"] = t["euclid_dist"].rank().astype(int)
    t["cosine_rank"] = t["cosine_sim"].rank(ascending=False).astype(int)
    return t
def clustering_scores(df):
    """Group all athletes into one cluster per sport, once per metric, and score the result."""
    X = df[PCT].values
    k = df["sport"].nunique()
    results = []
    for name, metric in [("Euclidean", "euclidean"), ("Cosine", "cosine")]:
        labels = AgglomerativeClustering(n_clusters=k, metric=metric,
                                         linkage="average").fit_predict(X)
        results.append({
            "Metric": name,
            "Silhouette": silhouette_score(X, labels, metric=metric),
            "Adjusted Rand vs true sports": adjusted_rand_score(df["sport"], labels),
        })
    return pd.DataFrame(results)
# ---------------- STEP 6: talent-transfer validation ----------------
# (source sport, destination sport, sex): known or plausible transfer pathways
TRANSFER_CASES = [
    ("Wrestling", "Kabaddi", "M"),
    ("Weightlifting", "Rowing", "M"),
    ("Sprinting", "LongJump", "M"),
    ("Swimming", "Rowing", "F"),
    ("Football", "Badminton", "F"),
    ("Judo", "Wrestling", "M"),
    ("Basketball", "Volleyball", "F"),
    ("ShotPut", "Weightlifting", "M"),
]


def raw_to_pct(raw, df, sex):
    """Turn one athlete's RAW test values into percentile scores,
    using the same-sex athletes in the CSV as the reference."""
    ref = df[df["sex"] == sex]
    vec = []
    for i, feat in enumerate(FEATURES):
        p = (ref[feat].values < raw[i]).mean()
        vec.append(1 - p if feat in LOWER_IS_BETTER else p)
    return np.array(vec)


def make_hybrid(df, source, dest, sex, blend, rng):
    """Create ONE fake transfer athlete: a mix of the source sport's profile and the
    destination sport's profile (blend=0.5 means half and half). Built only from the CSV."""
    g = df[df["sex"] == sex]
    gs = g[g["sport"] == source][FEATURES]
    gd = g[g["sport"] == dest][FEATURES]
    mean = (1 - blend) * gs.mean().values + blend * gd.mean().values
    spread = (1 - blend) * gs.std().values + blend * gd.std().values
    leftovers = pd.concat([gs - gs.mean(), gd - gd.mean()])
    corr = np.corrcoef(leftovers.values.T)
    sample = rng.multivariate_normal(mean, np.outer(spread, spread) * corr)
    return np.clip(sample, 0.1, None)


def transfer_results(df, blend=0.5, n_trials=30, seed=7):
    """For each transfer pair: does each metric rank the destination sport first?"""
    cents = sport_centroids(df)
    sports = sorted(df["sport"].unique())
    rng = np.random.default_rng(seed)
    out = []
    for source, dest, sex in TRANSFER_CASES:
        if source not in sports or dest not in sports:
            continue
        eu_hits = co_hits = 0
        eu_ranks, co_ranks = [], []
        for _ in range(n_trials):
            raw = make_hybrid(df, source, dest, sex, blend, rng)
            vec = raw_to_pct(raw, df, sex)
            t = compare_to_sports(vec, cents, sex)
            t = t[t["sport"] != source]                    # the source sport is not allowed as an answer
            eu_order = t.sort_values("euclid_dist")["sport"].tolist()
            co_order = t.sort_values("cosine_sim", ascending=False)["sport"].tolist()
            eu_hits += eu_order[0] == dest
            co_hits += co_order[0] == dest
            eu_ranks.append(eu_order.index(dest) + 1)
            co_ranks.append(co_order.index(dest) + 1)
        out.append({"Pair": f"{source} -> {dest} ({sex})",
                    "Euclid top-1": eu_hits / n_trials, "Cosine top-1": co_hits / n_trials,
                    "Euclid avg rank": np.mean(eu_ranks), "Cosine avg rank": np.mean(co_ranks)})
    return pd.DataFrame(out)
# ---------------- STEP 7b: improvement with unsupervised learning ----------------
def whitened_features(df):
    """PCA whitening: removes the overlap between correlated tests
    (sprint, jump and broad jump all measure leg power)."""
    return PCA(whiten=True, random_state=0).fit_transform(df[PCT].values)


def own_sport_accuracy(df, Z, metric, ks=(1, 3, 5)):
    """Share of athletes whose own sport is among the metric's top-k closest sports.
    Z is the feature matrix to use (one row per athlete). Compared within each sex."""
    y = df["sport"].to_numpy(dtype=str)
    hits = {k: 0 for k in ks}
    for sex in ("M", "F"):
        m = (df["sex"] == sex).values
        Zs, ys = Z[m], y[m]
        sports = np.array(sorted(set(ys)))
        C = np.array([Zs[ys == s].mean(axis=0) for s in sports])
        if metric == "euclidean":
            score = -((Zs[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        else:
            score = (Zs / np.linalg.norm(Zs, axis=1, keepdims=True)) @ \
                    (C / np.linalg.norm(C, axis=1, keepdims=True)).T
        order = np.argsort(-score, axis=1)
        for k in ks:
            hits[k] += (sports[order[:, :k]] == ys[:, None]).any(axis=1).sum()
    return {k: hits[k] / len(df) for k in ks}


def accuracy_table(df):
    """Own-sport accuracy for the basic metrics and the whitened improvement."""
    X, W = df[PCT].values, whitened_features(df)
    rows = []
    for name, Z, metric in [("Euclidean (percentiles)", X, "euclidean"),
                            ("Cosine (percentiles)", X, "cosine"),
                            ("Euclidean + PCA whitening", W, "euclidean")]:
        acc = own_sport_accuracy(df, Z, metric)
        rows.append({"Method": name, "Top-1": acc[1], "Top-3": acc[3], "Top-5": acc[5]})
    t = pd.DataFrame(rows)
    t.attrs["chance_top1"] = 1 / df["sport"].nunique()
    return t


def gmm_agreement(df):
    """Gaussian Mixture clustering (unsupervised): how well do its groups match the real sports?"""
    k = df["sport"].nunique()
    labels = GaussianMixture(n_components=k, random_state=0, n_init=2).fit_predict(df[PCT].values)
    return adjusted_rand_score(df["sport"], labels)
def pca_coords(df):
    """Squeeze the 7 percentile features into 2 numbers per athlete (for plotting)."""
    return PCA(n_components=2, random_state=0).fit_transform(df[PCT].values)