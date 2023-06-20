import cv2 
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm 
import pandas as pd
import numpy as np 

class SIRTFeatures():
  """Will return a csv of generated features. Receives a directory that contains images"""
  def __init__(self,image_folder_path, n_features = 100) -> None:
    self.path = image_folder_path
    self.filename = f'{image_folder_path}_descriptors'
    self.n_features = n_features
    self.descriptor_list = self.load_and_process(self.path)
    self.to_csv(self.descriptor_list)
  
  def load_and_process(self, path, valid_types=['.jpg', '.JPEG']):
    """load and process saves memory by only saving the descriptors and not the image itself
    """
    des_list = []
    list_dir = os.listdir(path)
    count = 0
    for f in tqdm.tqdm(list_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_types:
            continue
        image = cv2.imread(os.path.join(path, f))
        des = self.generate_descriptor(image)  # Assuming generate_descriptor function is defined elsewhere

        des_list.append(des)
        count +=1
    
    des_list = pd.DataFrame(des_list) 
    return des_list

  def generate_descriptor(self,image):
    # Convert the image to grayscale (SIFT works on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=self.n_features)
    # Find keypoints and dfcriptors directly
    kp, des = sift.detectAndCompute(gray, None)
    des = np.array(des).flatten()
    print(des.shape)
    
    return des
  
  def to_csv(self,df): 
    try:
      assert type(self.filename) == str
      df.to_csv(self.filename+'.csv')
    except AssertionError as a:
      print(a)
    return 0


SIRTFeatures(image_folder_path= 'test', n_features=100)

'''
def generate_kmeans(n_clusters, dataframe):
  kmeans = KMeans(n_clusters=4, n_init=10, max_iter=50, random_state=1)
  kmeans.fit(dataframe) #`fit_predict` entrena el modelo y devuelve las predicciones

  y_pred = kmeans.labels_ #kmeans.predict(rates)
  y_pred


  pca = PCA(n_components=2).fit(dataframe)
  rates_pca = pca.transform(dataframe) 
  #print(rates_pca)
  sns.scatterplot(x=rates_pca[:, 0], y=rates_pca[:, 1], hue=y_pred, palette='Set1')

  plt.title("Visualización PCA (n = 2) para 4 clusters por Kmeans")
  plt.show()
'''