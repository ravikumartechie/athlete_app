"""Athlete Talent Comparison App  --  run with:  py -m streamlit run app.py"""
import io
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import engine as E

st.set_page_config(page_title="Athlete Talent Comparison", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "athletes_dataset.csv")

# ---------------- sidebar: choose the dataset ----------------
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
st.sidebar.caption("Required columns: " + ", ".join(E.REQUIRED))
if uploaded is not None:
    csv_bytes, source_name = uploaded.getvalue(), uploaded.name
else:
    with open(DEFAULT_CSV, "rb") as fh:
        csv_bytes = fh.read()
    source_name = "athletes_dataset.csv (default)"
st.sidebar.caption("Using: " + source_name)


@st.cache_data
def prepare(raw_bytes):
    """Load the CSV, add the percentile columns, and compute each sport's centroid."""
    df = E.normalize_percentile(E.load_csv(io.BytesIO(raw_bytes)))
    return df, E.sport_centroids(df)


try:
    df, cents = prepare(csv_bytes)
except Exception as err:
    st.error(f"Could not load the dataset: {err}")
    st.stop()

sports = sorted(df["sport"].unique())
st.title("Cross-Discipline Athlete Talent Comparison")
st.caption(f"Euclidean vs Cosine similarity | {len(df):,} athletes | {len(sports)} sports")

tab_data, tab_norm, tab_match, tab_compare, tab_improve, tab_about = st.tabs(
    ["Dataset", "Normalization", "Athlete Match", "Metric Comparison", "Improvement", "About"])
# ---------------- tab 1: dataset ----------------
with tab_data:
    a, b, c, d = st.columns(4)
    a.metric("Athletes", f"{len(df):,}")
    b.metric("Sports", len(sports))
    c.metric("Female / Male", f"{(df.sex == 'F').sum()} / {(df.sex == 'M').sum()}")
    d.metric("Sport groups", df["sport_group"].nunique() if "sport_group" in df else "n/a")

    pick = st.multiselect("Filter sports", sports, default=sports)
    sub = df[df["sport"].isin(pick)]
    st.dataframe(sub[["sport", "sex"] + E.FEATURES], hide_index=True, height=280)

    st.markdown("**Average test results by sport**")
    st.dataframe(sub.groupby("sport")[E.FEATURES].mean().round(2))

    xy = E.pca_coords(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in pick:
        m = (df["sport"] == s).values
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, label=s)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA view of the athletes")
    ax.legend(fontsize=6, ncol=3)
    st.pyplot(fig)
# ---------------- tab 2: normalization (Objective 1) ----------------
with tab_norm:
    st.markdown("**Objective 1.** Every test becomes a **percentile (0 to 1) within the athlete's own sex**, "
                "pooled across all sports. Sprint and agility are flipped so higher is always better.")
    feat = st.selectbox("Test to inspect", E.FEATURES)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for s in ("M", "F"):
        ax[0].hist(df[df["sex"] == s][feat], bins=30, alpha=0.6, label=s)
        ax[1].hist(df[df["sex"] == s]["pct_" + feat], bins=30, alpha=0.6, label=s)
    ax[0].set_title("Raw values (the sexes differ)")
    ax[1].set_title("Percentile scores (common scale)")
    ax[0].legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Left: raw numbers. Men and women sit at different levels, and each test has its own units. "
               "Right: after normalization both sexes share the same 0 to 1 scale.")
# ---------------- tab 3: athlete match ----------------
with tab_match:
    st.markdown("Enter an athlete's test results. The app converts them to percentile scores "
                "(compared with athletes of the same sex), then finds the closest sport under each metric.")
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    ref = df[df["sex"] == sex]
    cols = st.columns(4)
    raw = []
    for i, f in enumerate(E.FEATURES):
        lo, hi = float(ref[f].min() * 0.8), float(ref[f].max() * 1.2)
        raw.append(cols[i % 4].number_input(f, lo, hi, float(round(ref[f].median(), 2))))

    vec = E.raw_to_pct(raw, df, sex)
    t = E.compare_to_sports(vec, cents, sex)
    eu = t.sort_values("euclid_dist")["sport"].iloc[0]
    co = t.sort_values("cosine_sim", ascending=False)["sport"].iloc[0]

    m1, m2 = st.columns(2)
    m1.metric("Best match: Euclidean", eu)
    m2.metric("Best match: Cosine", co)
    if eu != co:
        st.info("The metrics disagree. Euclidean reflects overall closeness (how strong the athlete is). "
                "Cosine reflects the shape of the profile (what type of athlete).")

    left, right = st.columns(2)
    left.dataframe(t.sort_values("euclid_rank").round(3), hide_index=True, height=420)
    with right:
        names = t["sport"].tolist()
        other = st.selectbox("Radar chart: compare with", names, index=names.index(co))
        ang = np.linspace(0, 2 * np.pi, len(E.FEATURES), endpoint=False).tolist()
        ang += ang[:1]
        fig = plt.figure(figsize=(4.6, 4.6))
        ax = plt.subplot(polar=True)
        for label, v in [("This athlete", vec), (other, cents[(other, sex)])]:
            vv = list(v) + [v[0]]
            ax.plot(ang, vv, label=label)
            ax.fill(ang, vv, alpha=0.15)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([x.replace("_", "\n") for x in E.FEATURES], fontsize=7)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        st.pyplot(fig)
# ---------------- tab 4: metric comparison (Objectives 2 and 3) ----------------
@st.cache_data
def cached_cluster(raw_bytes):
    return E.clustering_scores(prepare(raw_bytes)[0])


@st.cache_data
def cached_transfer(raw_bytes, blend):
    return E.transfer_results(prepare(raw_bytes)[0], blend=blend)


with tab_compare:
    st.markdown("**Objective 2.** Clustering quality of each metric")
    cs = cached_cluster(csv_bytes)
    st.dataframe(cs.round(3), hide_index=True)
    st.caption("Silhouette: tighter, better-separated clusters (higher is better). "
               "Adjusted Rand: agreement with the true sports (higher is better).")

    st.markdown("**Objective 3.** Talent-transfer validation")
    st.caption("Each transfer athlete is a blend of a source sport and a destination sport. "
               "A metric succeeds when it ranks the destination sport first.")
    blend = st.slider("Blend toward the destination sport (0.5 = half source, half destination)",
                      0.3, 0.8, 0.5, 0.05)
    tr = cached_transfer(csv_bytes, blend)
    st.dataframe(tr.round(2), hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(tr))
    w = 0.38
    ax.bar(x - w / 2, tr["Euclid top-1"], w, label="Euclidean")
    ax.bar(x + w / 2, tr["Cosine top-1"], w, label="Cosine")
    ax.set_xticks(x)
    ax.set_xticklabels(tr["Pair"], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 accuracy")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.success(f"Overall top-1: Euclidean {tr['Euclid top-1'].mean():.1%} | "
               f"Cosine {tr['Cosine top-1'].mean():.1%}")
# ---------------- tab 5: improvement (unsupervised learning) ----------------
@st.cache_data
def cached_accuracy(raw_bytes):
    d = prepare(raw_bytes)[0]
    return E.accuracy_table(d), E.gmm_agreement(d)


with tab_improve:
    acc, gmm_ari = cached_accuracy(csv_bytes)
    n_sports = len(sports)
    st.markdown("**Own-sport accuracy:** does the sport the athlete really plays appear among the "
                "metric's closest sports?")
    show = acc.copy()
    for col in ["Top-1", "Top-3", "Top-5"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    st.dataframe(show, hide_index=True)
    st.caption(f"Random guessing gives about {1 / n_sports:.0%} for top-1 and {5 / n_sports:.0%} for top-5.")

    st.markdown("**PCA whitening** (unsupervised) removes the overlap between correlated tests "
                "(sprint, jump and broad jump all measure leg power), which raises Euclidean accuracy.")
    st.metric("Gaussian Mixture clustering: Adjusted Rand vs true sports", f"{gmm_ari:.3f}")
    st.warning("Accuracy is capped by the data. Many sports have genuinely overlapping fitness profiles, "
               "so even a supervised model that sees the sport labels reaches only about 55% exact-sport "
               "accuracy on the default dataset.")

# ---------------- tab 6: about ----------------
with tab_about:
    st.subheader("Project objectives")
    st.markdown("""
1. Normalize cross-discipline fitness-test and performance-trial features to a common scale.
2. Compare Euclidean and Cosine similarity for cross-discipline athlete-profile comparison.
3. Validate the metric choice against cross-discipline talent-transfer cases.
""")
    st.subheader("Data honesty")
    st.markdown("""
No public per-athlete cross-sport fitness dataset exists, so the default CSV is **synthetic**. Some test
means are anchored to published ranges, the rest are estimated from each sport's physical demands, and the
sex multipliers, age band and training effect are assumptions. Transfer athletes are simulated blends of
two sports. The app accepts any CSV with the required columns, so real data can replace the synthetic file
without changing any code.
""")