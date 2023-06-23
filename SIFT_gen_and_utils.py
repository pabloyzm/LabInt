import cv2 
import os
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm 
import pandas as pd
import numpy as np   
from sklearn.model_selection import train_test_split

class SIFTFeatures():
  """Will return a csv of flattened descriptors. Receives a directory that contains images, ndescriptors, subsample ratio"""
 
  def __init__(self,image_folder_path, n_features = 300, subsample_ratio =1) -> None:
    self.path = image_folder_path
    self.filename = f'{image_folder_path}_descriptors'
    self.n_features = n_features
    self.subsample_ratio = subsample_ratio
    self.descriptors_list = self.load_and_process(self.path)
    
  def load_and_process(self, path, valid_types=['.jpg','.JPG','.jpeg' ,'.JPEG']):
    """load and process saves memory by only saving the descriptors and not the image itself
    """
    descriptors_for_kmeans_list = []
    descriptors_for_quantization_list = []

    list_dir = os.listdir(path)
    if self.subsample_ratio != 1:
      list_dir_for_kmeans, list_dir_for_quantization = train_test_split(list_dir, train_size=self.subsample_ratio)#,random_state=1)
    count = 0

    for f in tqdm.tqdm(list_dir_for_kmeans):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_types:
            continue
        image = cv2.imread(os.path.join(path, f))
        des_kmeans = self.generate_descriptor(image)  # Assuming generate_descriptor function is defined elsewhere 
        descriptors_for_kmeans_list.append(des_kmeans)
        count +=1
    descriptors_for_kmeans_list = pd.DataFrame(descriptors_for_kmeans_list) 
    self.to_csv(descriptors_for_kmeans_list,subindex='kmeans')
    del descriptors_for_kmeans_list

    for f in tqdm.tqdm(list_dir_for_quantization):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_types:
            continue
        image = cv2.imread(os.path.join(path, f))
        des_quant = self.generate_descriptor(image)  # Assuming generate_descriptor function is defined elsewhere 
        descriptors_for_quantization_list.append(des_quant)
        count +=1
    descriptors_for_quantization_list = pd.DataFrame(descriptors_for_quantization_list)
    self.to_csv(descriptors_for_quantization_list,subindex='quant')
    return 0
    
  def generate_descriptor(self,image):
    # Convert the image to grayscale (SIFT works on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=self.n_features)
    # Find keypoints and dfcriptors directly
    kp, des = sift.detectAndCompute(gray, None)
    #del kp
    des = np.array(des).flatten()  
    #print(kp)
    return des
  
  def to_csv(self,df,subindex = ''):  
   
    try:
      assert type(self.filename) == str
      print("saving df as csv file...")
      df.to_csv(self.filename+'_'+subindex+'.csv',index = False)
      print("Finished saving.")
      del df
    except AssertionError as a:
      print(a)
    return 0


def initialize_kmeans(dataframe_kmeans, elbow = True, plot = False):
  """to calibrate de numer of clusters, use elbow method"""
  if elbow:
    SSE = []
    numClusters = list(range(1, 7))

    for k in tqdm(numClusters):
        k_means = KMeans(n_clusters=k, n_init=10, random_state=1)
        k_means.fit(dataframe_kmeans)
        SSE.append(k_means.inertia_)

    plt.plot(numClusters, SSE, marker="o")
    plt.title('Método del codo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('SSE')
    plt.show()

  n_clusters = int(input('enter number of kmeans to compute:'))
  assert type(n_clusters) == int
  #actual kmeans
  kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, max_iter=15, random_state=1)
  kmeans.fit(dataframe_kmeans) #`fit_predict` entrena el modelo y devuelve las predicciones
  if plot:
    y_pred = kmeans.labels_ #kmeans.predict(rates)
    y_pred 
    
    pca = PCA(n_components=2).fit(dataframe_kmeans)
    rates_pca = pca.transform(dataframe_kmeans) 
    #print(rates_pca)
  # plt.figure(figsize=(10, 10))
    sns.scatterplot(x=rates_pca[:, 0], y=rates_pca[:, 1], hue=y_pred, palette='Set1',markers='x')
    plt.title(f"Visualización PCA (n = 2) para {n_clusters} clusters por Kmeans")
    plt.show() 

  #visual_word_assignments = kmeans.predict(for_quatinzation)
  #histogram = np.zeros(n_clusters, dtype=np.int32)

  # Count the occurrences of each visual word in the descriptors
  #for assignment in visual_word_assignments:
    #  histogram[assignment] += 1

  #print('Histogram:', histogram)
  #return histogram

##runtime 
SIFTFeatures(image_folder_path= 'scripts/LabInt/test', n_features=300, subsample_ratio = 0.2)

 