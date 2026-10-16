"""
Builds athletes_dataset.csv: 4000 synthetic athletes, 20 sports, both sexes, 7 fitness tests.

DATA HONESTY: no public per-athlete cross-sport fitness dataset exists, so this is SYNTHETIC.
 [SOURCED]   Some VO2max, vertical-jump and grip means are anchored to published ranges.
 [ESTIMATED] Other means are set from each sport's known physical demands.
             Test-to-test correlations follow Enoksen et al. 2011.
 [ASSUMED]   Sex multipliers, the 14-23 age band, the training-years effect.
"""
import numpy as np
import pandas as pd

SEED = 42
N_PER_GROUP = 100     # athletes per sport per sex  (20 sports x 2 sexes x 100 = 4000)

FEATURES = ["sprint_30m_s", "vertical_jump_cm", "broad_jump_cm", "vo2max",
            "grip_kg", "agility_s", "flexibility_cm"]
LOWER_IS_BETTER = {"sprint_30m_s", "agility_s"}     # times: smaller = better

# sport: (group, [(mean, spread) for each of the 7 tests, MALE values])
SPORTS = {
 "Sprinting":      ("Speed-Power",  [(4.05,.12),(55,5),(285,15),(55,4),(46,5),(9.7,.25),(28,5)]),
 "LongJump":       ("Speed-Power",  [(4.15,.12),(59,5),(305,15),(52,4),(44,5),(9.8,.25),(30,5)]),
 "Weightlifting":  ("Strength",     [(4.85,.20),(52,6),(265,15),(43,4),(62,7),(10.5,.35),(20,5)]),
 "ShotPut":        ("Strength",     [(4.75,.20),(50,6),(280,18),(40,4),(60,7),(10.4,.35),(22,5)]),
 "Swimming":       ("Endurance",    [(4.75,.20),(45,5),(250,15),(58,4),(44,5),(10.4,.30),(36,5)]),
 "Rowing":         ("Endurance",    [(4.85,.20),(40,5),(245,15),(65,4),(50,6),(10.7,.35),(30,5)]),
 "DistanceRunning":("Endurance",    [(4.35,.15),(50,5),(265,15),(72,4),(42,5),(9.8,.25),(28,5)]),
 "Cycling":        ("Endurance",    [(4.45,.15),(48,5),(260,15),(68,4),(44,5),(10.0,.28),(26,5)]),
 "Football":       ("Team",         [(4.25,.15),(42,5),(265,15),(56,4),(43,5),(9.5,.25),(27,5)]),
 "Hockey":         ("Team",         [(4.25,.15),(50,5),(270,15),(56,4),(44,5),(9.4,.25),(28,5)]),
 "Basketball":     ("Team",         [(4.30,.15),(62,6),(275,15),(54,4),(46,5),(9.5,.25),(27,5)]),
 "Volleyball":     ("Team",         [(4.40,.15),(63,6),(272,15),(51,4),(45,5),(9.6,.25),(26,5)]),
 "Kabaddi":        ("Team",         [(4.40,.16),(44,6),(258,15),(50,4),(47,6),(9.3,.22),(26,5)]),
 "Cricket":        ("Team",         [(4.45,.18),(44,5),(255,15),(48,4),(46,6),(9.9,.28),(26,5)]),
 "Boxing":         ("Combat",       [(4.45,.18),(48,5),(262,15),(52,4),(48,5),(9.6,.28),(27,5)]),
 "Wrestling":      ("Combat",       [(4.55,.18),(46,6),(260,15),(51,4),(54,6),(9.9,.30),(24,5)]),
 "Judo":           ("Combat",       [(4.50,.18),(46,6),(255,15),(52,4),(58,6),(9.9,.30),(27,5)]),
 "Badminton":      ("Racquet-Skill",[(4.35,.15),(47,5),(268,15),(53,4),(42,5),(9.4,.22),(32,5)]),
 "Tennis":         ("Racquet-Skill",[(4.35,.15),(47,5),(265,15),(55,4),(43,5),(9.4,.22),(30,5)]),
 "Gymnastics":     ("Racquet-Skill",[(4.20,.15),(58,5),(280,15),(54,4),(50,6),(9.4,.22),(42,6)]),
}

# Female value = male value x ratio (same order as FEATURES)
SEX_RATIO = dict(zip(FEATURES, [1.095, .78, .78, .87, .65, 1.08, 1.10]))

# How the 7 tests relate to each other (e.g. fast sprinters jump higher)
CORR = np.array([
 [ 1.00,-0.55,-0.50,-0.10,-0.15, 0.50, 0.00],
 [-0.55, 1.00, 0.75,-0.10, 0.25,-0.40, 0.10],
 [-0.50, 0.75, 1.00,-0.10, 0.25,-0.35, 0.10],
 [-0.10,-0.10,-0.10, 1.00,-0.20,-0.15, 0.05],
 [-0.15, 0.25, 0.25,-0.20, 1.00, 0.10,-0.05],
 [ 0.50,-0.40,-0.35,-0.15, 0.10, 1.00,-0.05],
 [ 0.00, 0.10, 0.10, 0.05,-0.05,-0.05, 1.00]])


def build():
    rng = np.random.default_rng(SEED)
    rows = []
    for sport, (group, params) in SPORTS.items():
        for sex in ("M", "F"):
            for i in range(N_PER_GROUP):
                years = rng.uniform(1, 10)
                age = rng.uniform(14, 23)
                boost = 1 + 0.008 * (years - 5)          # more training = slightly better
                means, spreads = [], []
                for feat, (m, s) in zip(FEATURES, params):
                    if sex == "F":
                        m, s = m * SEX_RATIO[feat], s * SEX_RATIO[feat]
                    means.append(m / boost if feat in LOWER_IS_BETTER else m * boost)
                    spreads.append(s)
                spreads = np.array(spreads)
                cov = np.outer(spreads, spreads) * CORR
                values = np.clip(rng.multivariate_normal(means, cov), 0.1, None)
                row = {"athlete_id": f"{sport[:4].upper()}_{sex}_{i:03d}",
                       "sport": sport, "sport_group": group, "sex": sex,
                       "age": round(age, 1), "years_training": round(years, 1)}
                row.update({f: round(v, 2) for f, v in zip(FEATURES, values)})
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv("athletes_dataset.csv", index=False)
    print(df.shape, "| sports:", df.sport.nunique(), "| groups:", df.sport_group.nunique())