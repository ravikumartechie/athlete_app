import engine as E

df = E.normalize_percentile(E.load_csv("athletes_dataset.csv"))

print("Rows:", len(df))
print()
print("Raw sprint time, average by sex:")
print(df.groupby("sex")["sprint_30m_s"].mean().round(2))
print()
print("Percentile score, average by sex (should be about equal):")
print(df.groupby("sex")["pct_sprint_30m_s"].mean().round(2))
print()
print("Best and worst average sprint percentile by sport:")
avg = df.groupby("sport")["pct_sprint_30m_s"].mean().sort_values(ascending=False)
print(avg.head(3).round(2))
print(avg.tail(3).round(2))