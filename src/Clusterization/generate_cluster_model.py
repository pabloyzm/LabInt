#
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# import seaborn as sns
# from sklearn.cluster import MiniBatchKMeans
# from tqdm import tqdm
# import pandas as pd
#
# def initialize_kmeans(kmeans_csv_path:str, elbow = True, elbow_vector  = np.linspace(10,1000,10, dtype=int) ,plot = False):
#     """to calibrate de numer of clusters, use elbow method"""
#     print(f'Initializing MiniBatchKmeans')
#     kmeans_df = pd.read_csv(f'{kmeans_csv_path}/images_descriptors_kmeans.csv')
#     kmeans_df = kmeans_df.drop(columns = ['image_name', 'class'])
#     #kmeans_df = np.reshape(kmeans_df,(len(kmeans_df)*for_kmeans.iloc[0].shape[0]//128,128))
#     #quatization_df = np.reshape(quatization_df,(len(quatization_df)*for_quatinzation.iloc[0].shape[0]//128,128))
#     if elbow:
#         SSE = []
#         numClusters =  elbow_vector
#
#         for k in tqdm(numClusters):
#             k_means = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=1, batch_size=512)
#             k_means.fit(kmeans_df)
#             SSE.append(k_means.inertia_)
#
#         plt.plot(numClusters, SSE, marker="o")
#         plt.title('Método del codo')
#         plt.xlabel('Número de Clusters')
#         plt.ylabel('SSE')
#         plt.show()
#
#     # choose the best number of clusters based on the variation of the SSE
#     variation = [(SSE[i] - SSE[i+1])/ SSE[i] * 100 for i in range(len(SSE)-1)]
#     n_clusters = numClusters[variation.index(max(variation)) + 1]
#     print(f"El número óptimo de clusters es {n_clusters}")
#
#     #actual kmeans
#     kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=1)
#     kmeans.fit(kmeans_df) #`fit_predict` entrena el modelo y devuelve las predicciones
#     if plot:
#         y_pred = kmeans.labels_ #kmeans.predict(rates)
#         pca = PCA(n_components=2).fit(kmeans_df)
#         rates_pca = pca.transform(kmeans_df)
#         plt.figure(figsize=(10, 10))
#         sns.scatterplot(x=rates_pca[:, 0], y=rates_pca[:, 1], hue=y_pred, palette='Set1',markers='x')
#         plt.title(f"Visualización PCA (n = 2) para {n_clusters} clusters por Kmeans")
#         plt.show()
#     return kmeans, n_clusters


import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import pandas as pd

def initialize_kmeans(kmeans_csv_path:str, chunk_size:int = 50000, batch_size= 4096,elbow = True, elbow_vector  = np.linspace(10,1000,10, dtype=int) ,plot = False):
    """to calibrate de numer of clusters, use elbow method"""
    print(f'Initializing MiniBatchKmeans')
    SSE = []
    if elbow:
        numClusters =  elbow_vector
        for k in tqdm(numClusters):
            k_means = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=1, batch_size=batch_size)
            chunk_iter = pd.read_csv(f'{kmeans_csv_path}/images_descriptors_kmeans.csv', chunksize=chunk_size)
            for chunk in chunk_iter:
                chunk = chunk.drop(columns = ['image_name', 'class'])
                k_means.partial_fit(chunk)
            SSE.append(k_means.inertia_)
        plt.plot(numClusters, SSE, marker="o")
        plt.title('Método del codo')
        plt.xlabel('Número de Clusters')
        plt.ylabel('SSE')
        plt.show()

    # choose the best number of clusters based on the variation of the SSE
    variation = [(SSE[i] - SSE[i+1])/ SSE[i] * 100 for i in range(len(SSE)-1)]
    n_clusters = numClusters[variation.index(max(variation)) + 1]
    print(f"El número óptimo de clusters es {n_clusters}")

    #actual kmeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=1)
    chunk_iter = pd.read_csv(f'{kmeans_csv_path}/images_descriptors_kmeans.csv', chunksize=chunk_size)
    for chunk in chunk_iter:
        chunk = chunk.drop(columns = ['image_name', 'class'])
        kmeans.partial_fit(chunk)
    if plot:
        chunk_iter = pd.read_csv(f'{kmeans_csv_path}/images_descriptors_kmeans.csv', chunksize=chunk_size)
        pca = PCA(n_components=2).fit_transform(pd.concat(chunk_iter))
        y_pred = kmeans.predict(pca)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=y_pred, palette='Set1',markers='x')
        plt.title(f"Visualización PCA (n = 2) para {n_clusters} clusters por Kmeans")
        plt.show()
    return kmeans, n_clusters
