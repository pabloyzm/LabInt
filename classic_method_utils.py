#import numpy as np
#aqui posibles utils en el futuro que sean demasiado grandes pa poner 
#en el notebook comodamente
import os , os.path
from PIL import Image
from skimage.feature import local_binary_pattern
import numpy as np

def classic_method(image):
    image = np.array(image)
    #print(image.shape)
    "Implements LBP algorithm "
    lbp = local_binary_pattern(image, P=8, R=1)
    #segmentar
    
    M = (lbp.shape[0])#//10
    N = (lbp.shape[1])#//10
    tiles = [lbp[x:x+M,y:y+N] for x in range(0,lbp.shape[0],M) for y in range(0,lbp.shape[1],N)]
        
    #segment histogramas  
    tiles_histogram = []
    #calcular el histograma para cada segmento
    for i in (tiles): 
        hist = np.histogram(i,density=True,bins = 59,range=(0, 59))[0]
        tiles_histogram.append(hist)
    full_histogram = np.array(tiles_histogram).flatten()
    return np.array(full_histogram) 

def padding(image):
     width = image.shape
     height = image.shape
 

def load_and_process(paths = ['data/GPR1200/images/'], valid_types = [".jpg",".JPEG"]):
    """ load and process saves memory by only saving the vectorized image and not the image itself"""
    hists = []

    for i in range(len(paths)):
        list_dir = os.listdir(paths[i])
        
        for f in list_dir:
            list_dir.set_description(f"Loading images from directory {paths[i]}")
            ext = os.path.splitext(f)[1]
            
            if ext.lower() not in valid_types:
                continue
            image = (Image.open(os.path.join(paths[i],f)).convert('L'))
            hist = classic_method(image)
            hists.append(hist)
    return hists

