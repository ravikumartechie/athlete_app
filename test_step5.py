import engine as E

df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))
print("Clustering 4000 athletes into 20 groups, please wait...")
print()
print(E.clustering_scores(df).round(3).to_string(index=False))