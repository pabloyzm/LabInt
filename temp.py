import cv2
import time

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# Load an image


start = time.time()
img = cv2.imread('test1.jpg')

# Convert the image to grayscale (SIFT works on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and dfcriptors directly
kp, des = sift.detectAndCompute(gray, None)
end = time.time()
dt = end-start
print(dt)

# Print descriptors
print('Descriptors:', des)

print(des.shape)
# If you also want to visualize the keypoints in your image, use the following line
#img = cv2.drawKeypoints(gray, kp, img)
#cv2.imshow('SIFT keypoints', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


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