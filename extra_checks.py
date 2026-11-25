"""Extra checks quoted in the paper. Run in the folder that has engine.py and athletes_dataset.csv:
    py extra_checks.py
Takes about a minute."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import engine as E

df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))

# 1) Run-to-run variation of the transfer test (10 different hybrid-sampling seeds)
eu, co = [], []
for seed in range(1, 11):
    r = E.transfer_results(df, seed=seed)
    eu.append(r["Euclid top-1"].mean() * 100)
    co.append(r["Cosine top-1"].mean() * 100)
eu, co = np.array(eu), np.array(co)
d = co - eu
print(f"Euclidean: {eu.mean():.1f}% (s.d. {eu.std(ddof=1):.1f})")
print(f"Cosine:    {co.mean():.1f}% (s.d. {co.std(ddof=1):.1f})")
print(f"Cosine ahead in {(d > 0).sum()} of 10 runs, mean difference {d.mean():.1f} points (s.d. {d.std(ddof=1):.1f})")

# 2) Supervised reference: random forest on the raw tests plus sex, 5-fold cross-validation
X = df[E.FEATURES].copy()
X["sex"] = (df["sex"] == "M").astype(int)
acc = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, df["sport"], cv=5).mean()
print(f"Random forest, 5-fold exact-sport accuracy: {acc * 100:.1f}%")
