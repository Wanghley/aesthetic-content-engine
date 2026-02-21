from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def cluster_life_events(df, n_clusters=12): # Roughly 1 cluster per month or major event
    embeddings = np.array(df['embedding'].tolist())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    df['cluster'] = kmeans.labels_
    
    # Selection Logic: In each cluster, find the top 5 highest aesthetic scores
    best_of_year = df.sort_values(['cluster', 'aesthetic_score'], ascending=[True, False]).groupby('cluster').head(5)
    return best_of_year