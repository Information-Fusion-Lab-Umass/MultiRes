import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

datapath = "Data"
filename =  "37_features(80-10-10).csv"     # 37_features(60-40-40).csv is renamed from "fast_slow_scores.csv"
filepath = os.path.join(datapath, filename)
plt.rcParams.update({'font.size': 15})

data = pd.read_csv(filepath).values #.to_numpy()

# Remove IDs
data = np.delete(data, np.s_[-1:], axis=1)

# Average over patients (i.e. rows)
#print (data[0].shape)
# feature_scores = np.average(data, axis=0)
feature_scores = np.sum(data, axis=0)
#print (feature_scores.shape)

# Save indices of sorted array
sorted_indices = np.argsort(feature_scores)
sorted_scores = feature_scores[sorted_indices]

# Plot histogram
x = np.arange(len(sorted_scores))
print(sorted_scores)
print(feature_scores)

# Scatter Plot
plt.plot(sorted_scores, marker='o')

# Annotate all points in the plot with feature number and score
for i, score in enumerate(sorted_scores):
    score = np.around(score, decimals=3)
    feature = int(np.where(feature_scores == sorted_scores[i])[0]) + 1   # Add 1 to remove 0 indexing
    label = 'Feature '+str(feature)+', '+str(score)
    plt.annotate(label, (x[i], sorted_scores[i]))


# fig, ax = plt.subplots()
# plot = ax.bar(x, sorted_scores)
#ax.set_xticklabels(np.around(sorted_scores, decimals=4))

plt.xlabel('Sorted Features (37)')
plt.ylabel('Fast-Slow Scores per feature')
plt.title('Scores per feature, averaged over all patients')
plt.grid(True)
plt.savefig('Plots/Scores per Feature - Bar Chart.png')
plt.show()