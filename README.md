Cross-Discipline Distance-Metric Study for Fair Athlete Talent Comparison Across Sports

Author: Ravi Kumar (Reg. No. 12401972), School of Computer Science & Engineering, Lovely Professional University
Live app: https://athleteapp-bbr3xwepgwef6zbwoxj2hz.streamlit.app/

What this project does

Athletes from different sports are tested on different scales: seconds, centimetres and kilograms. For sprint and agility a lower number is better, and men and women have different raw levels. This project:

Converts every fitness test to a percentile within each sex, so all tests share one 0 to 1 scale.
Compares Euclidean distance with cosine similarity for matching athlete profiles to sports.
Tests both metrics on simulated talent-transfer cases, such as Wrestling to Kabaddi.
Tries PCA whitening and Gaussian mixture clustering as unsupervised improvements.
Provides a six-tab Streamlit app to explore everything.
Important note about the data

The dataset is synthetic. No public per-athlete cross-sport fitness dataset exists, so 4,000 athletes (20 sports, both sexes, 7 fitness tests) were generated. Sport averages, sex ratios and correlations between tests are modelling assumptions. This is a methods study, not a validated tool for real athlete selection. The app accepts any CSV with the required columns, so real data can replace the synthetic file.

Files
File	Purpose
app.py	Streamlit web application (six tabs)
engine.py	All calculations: normalization, Euclidean and cosine, clustering, transfer test, whitening
athletes_dataset.csv	The synthetic dataset (4,000 athletes)
dataset_builder.py	Script that generates the dataset
make_figures.py	Creates the report figures in figures/
extra_checks.py	Ten-run comparison and random-forest reference result
requirements.txt	Python packages needed
How to run
py -m pip install -r requirements.txt
py -m streamlit run app.py

To use your own data, upload a CSV in the app sidebar. Required columns: sport, sex (M or F), sprint_30m_s, vertical_jump_cm, broad_jump_cm, vo2max, grip_kg, agility_s, flexibility_cm. A sport_group column is optional.

Main results (synthetic data)
Measure	Euclidean	Cosine
Transfer top-1 accuracy (one run)	50.4%	52.1%
Transfer top-1 accuracy (mean of 10 runs)	51.8%	54.0%
Silhouette score	0.089	0.111
Adjusted Rand index	0.158	0.090
Cosine was ahead in all 10 repeated transfer runs, by 2.3 points on average. The gap is small.
PCA whitening raised Euclidean own-sport top-1 accuracy from 49.2% to 55.5%.
Chance level is about 5% for 20 sports.
Limitations
All data are synthetic, and every mean, sex ratio and correlation is an assumption.
Transfer athletes are simulated blends, and the eight pairs were chosen by the author.
Only one dataset was generated, so repeated runs do not capture variation between datasets.
Own-sport accuracy is in-sample, because sport centroids include the tested athlete.
