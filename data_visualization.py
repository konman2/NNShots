import numpy as np
import matplotlib.pyplot as plt

x = np.load("./data/X.npy")[:50000]
y = np.load("./data/y.npy").T[:50000]

num_features = x.shape[1]
sp_feature = []
feature_bins = []
bin_frequency = []
for i in range(num_features):

    feature = x[:, i]
    _, feature_bin = np.histogram(feature)
    feature_bins.append(feature_bin)

    binned_indices = np.digitize(feature.flatten(), feature_bin)
    num_bins = np.max(binned_indices)
    sp_bin = []
    freq_bin = []
    for j in range(1, num_bins+1):
        freq_bin.append(np.sum(binned_indices == j))
        sp_bin.append(np.mean(y[(binned_indices == j)]))
    sp_feature.append(sp_bin)
    bin_frequency.append(freq_bin)

i = 0
titles = ['FINAL_MARGIN', 'SHOT_NUMBER', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'CLOSE_DEF_DIST']
for i in range(num_features):
    plt.title(titles[i])
    plt.plot(feature_bins[i], sp_feature[i])
    plt.xlabel(str(bin_frequency[i]))
    plt.show()