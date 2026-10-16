import engine as E

df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))
res = E.transfer_results(df)

print(res.round(2).to_string(index=False))
print()
print("Overall top-1 accuracy:")
print(f"Euclidean: {res['Euclid top-1'].mean():.1%}")
print(f"Cosine:    {res['Cosine top-1'].mean():.1%}")