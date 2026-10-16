import engine as E

df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))
cents = E.sport_centroids(df)

# Test A: does each metric pick the athlete's OWN sport as the closest one?
hits_eu = hits_co = 0
for _, row in df.iterrows():
    vec = row[E.PCT].values.astype(float)
    t = E.compare_to_sports(vec, cents, row["sex"])
    hits_eu += t.sort_values("euclid_rank").iloc[0]["sport"] == row["sport"]
    hits_co += t.sort_values("cosine_rank").iloc[0]["sport"] == row["sport"]

print("Test A: does the metric pick the athlete's OWN sport as closest?")
print(f"Euclidean: {hits_eu / len(df):.1%}")
print(f"Cosine:    {hits_co / len(df):.1%}")

# Test B: a sprinter's profile shape, but at only half the strength
sprinter_centre = cents[("Sprinting", "M")]
t = E.compare_to_sports(sprinter_centre * 0.5, cents, "M")
print()
print("Test B: sprinter profile at half strength")
print("Euclidean says:", t.sort_values("euclid_rank").iloc[0]["sport"])
print("Cosine says:   ", t.sort_values("cosine_rank").iloc[0]["sport"])