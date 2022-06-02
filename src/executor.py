### Executor contains the pipeline for the entire clustering flow, from preprocessing to modelling

import pandas as pd

from .preprocessing import preprocessing
from .utility import merge_cluster_labels
from .dbscan import DBSCANModel
from .kmeans import KMeansModel

def executor(df,model='kmeans', max_clusters=None, min_clusters=None, eps=None):
    data_preprocessed = preprocessing(df)

    if model == 'kmeans':
        print('Executing KMeans Clustering...')
        kmeans = KMeansModel(data_preprocessed) # instantiate kmeans model
        kmeans_models = kmeans.kmeans_model(min_clusters=1, max_clusters=10) # run multiple iterations of kmeans model

        optimal_clusters = kmeans.search_optimal_clusters() # search for optimal number of clusters

        cluster_labels, optimal_kmeans = kmeans.optimal_model()

        data_merged = merge_cluster_labels(data_preprocessed, cluster_labels)


    elif model == 'dbscan':
        print('Executing DBSCAN Clustering...')
        # define min_points
        MinPts = len(data_preprocessed.columns)*2 # MinPts should follow attributes*2

        # instantiate dbscan model
        dbscan  = DBSCANModel(data_preprocessed)
        cluster_labels, labels = dbscan.dbscan_model(eps=eps, min_samples= MinPts)
        optimal_eps = dbscan.search_optimal_minpts(MinPts)

        data_merged = merge_cluster_labels(data_preprocessed, cluster_labels)

    return print(data_merged.head())



if __name__ == "__main__":
    df = pd.read_csv("./data/CC_GENERAL.csv")

    executor(df)