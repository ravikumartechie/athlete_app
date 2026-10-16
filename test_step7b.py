import engine as E

df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))
acc = E.accuracy_table(df)

print(acc.round(3).to_string(index=False))
print()
print("Chance for top-1:", acc.attrs["chance_top1"])
print("Gaussian Mixture Adjusted Rand:", round(E.gmm_agreement(df), 3))