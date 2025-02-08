"""Image descriptor clustering for VGG (PCA + HBDSCAN)
"""

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN
import pandas as pd
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation, IncrementalPCA



tab_file_path = "./Histrogram_test_luminance/mit5k_big_vgg19_train.csv"

metric = "euclidean"

pd_dataset = pd.read_csv(tab_file_path)

pd_dataset_cluster = pd_dataset.drop(columns=['image_name', 'image_idx'])

names_column = list(pd_dataset_cluster)


X_clu = pd_dataset_cluster.values
X_clu = MinMaxScaler().fit_transform(X_clu)


min_cluster_size = int(0.01*len(X_clu))
max_cluster_size = int(0.05*len(X_clu))
cluster_interval = int((max_cluster_size - min_cluster_size)/20)

print("min_cluster_size",min_cluster_size)
print("max_cluster_size",max_cluster_size)
print("cluster_interval",cluster_interval)


parameters = {'min_cluster_size': np.arange(min_cluster_size, max_cluster_size, cluster_interval),
                  'variance_threshold': np.arange(5,20,1)
                    }

best_sil = 0.0
best_min_sample = 0
best_var_ther = 0.0

for params in product(*parameters.values()):

    sil_score = 0.0
    min_cluster_size = params[0]
    variance_threshold = params[1]
    print("min_cluster_size, variance_threshold", min_cluster_size, variance_threshold)
        
    try:
        X = PCA(n_components=variance_threshold).fit_transform(X_clu) #IncrementalPCA
        print(len(X[0]))
        hdb = HDBSCAN( min_cluster_size = min_cluster_size, metric = metric).fit(X)
        labels = hdb.labels_
        sil_score = metrics.silhouette_score(X, labels)
        try:
            sil_score = metrics.silhouette_score(X, labels)
            print("sil_score", sil_score)
            if sil_score >= best_sil:
                best_sil = sil_score
                best_min_sample = min_cluster_size
                best_var_ther = variance_threshold
        except:
            pass
    except:
        pass
        
    
    


print("best_sil =", best_sil)
print("best_min_sample =", best_min_sample)
print("best_var_ther =", best_var_ther)


X = PCA(n_components=best_var_ther).fit_transform(X_clu)
hdb = HDBSCAN(min_cluster_size = best_min_sample , metric= metric).fit(X)
labels = hdb.labels_

pd_dataset["labels"] = labels

pd_dataset.to_csv("./Histrogram_test_luminance/mit5k_big_vgg19_train_labels.csv", index=False)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print(f"Estimated number of clusters: {n_clusters_}")



