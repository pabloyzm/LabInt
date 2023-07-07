import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from src.SIFT.SIFT_gen_and_utils import SIFTFeatures
SIFT = SIFTFeatures("", "", run=False)
from tqdm import tqdm

def similarity_metric(vec_1,vec_2, measure = 'euclidean'):
    if measure == 'euclidean':
        resta = vec_1 - vec_2
        return np.sqrt(np.sum(resta**2))
    elif measure == 'manhattan':
        resta = vec_1 - vec_2
        return np.sum(np.abs(resta))
    elif measure == 'cosine':
        dot_product = np.dot(vec_1, vec_2)
        norm_a = np.linalg.norm(vec_1)
        norm_b = np.linalg.norm(vec_2)
        cosine_similarity = dot_product / (norm_a * norm_b)
        return 1 - cosine_similarity
    else:
        raise ValueError(f"Unknown measure: {measure}")

def query_image(df, image_name, measure="euclidean", feature_type = "CNN"):
    
    df["similarity"] = df[f"features_{feature_type}"].apply(lambda x: similarity_metric(x, df[df["image_name"] == image_name][f"features_{feature_type}"].values[0], measure=measure))
    df = df.sort_values(by=['similarity'])
    df = df.reset_index(drop=True)
    return df


def plot_10(df_, path_to_images):
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=(15, 6))
    fig.suptitle("Top Matching Images", fontsize=15)
    ax[0][0].set_title(f"Consulta")
    x = 0
    for j in range(0, 2):
        for i in range(0,5): 
            ax[j][i].set_axis_off()
            ax[j][i].imshow(Image.open(f'{path_to_images}/{df_["image_name"][x]}'))
            if x > 0:
                ax[j][i].set_title(f"Match {x-1}")
            x +=1
    plt.show()

def evaluate_query(df, image_query, measure="cosine", normalized=True, feature_type = "histogram"):
    # obtener la clase de la imagen
    
    image_class = SIFT.get_class(image_query)
    print(image_class)
    # calcular similudes
    df = query_image(df, image_query, measure=measure, feature_type = feature_type)
    # obtener los indices de las imagenes de la misma clase
    indexes = df[df["class"] == image_class].index.values.tolist()
    print(indexes)
    n_rel = len(indexes) - 1

    if normalized:
        N = len(df)
        return 1/(n_rel*N) * (sum(indexes) - (n_rel*(n_rel+1))/2)

    return 1/n_rel * sum(indexes)

def get_hist_from_str(value, separator):
    hist = str(value).strip().replace('[','').replace(']','').replace("\n","").split(separator)
    hist = [i for i in hist if i != '']
    hist = np.array(hist, dtype=np.float32)
    return hist

def query_image_normal(df_, image_name, measure="cosine", feature_type = "CNN"):
    df_["similarity"] = df_[f"features_{feature_type}"].apply(lambda x: similarity_metric(x, df_[df_["image_name"] == image_name][f"features_{feature_type}"].values[0], measure=measure))
    df_ = df_.sort_values(by=['similarity'])
    df_ = df_.reset_index(drop=True)
    return df_

def plot_histogram(df, index, n_bins, feature_type = 'histogram'):
    if feature_type == 'histogram':
        hist = get_hist_from_str(df[f"features_histogram"][index], separator=' ')
        x = np.arange(0, n_bins)
        plt.bar(x, hist)
        plt.title("Histogram")
        plt.xlabel("Quantized Value")
        plt.ylabel("Frequency")
        plt.show()
        return

    if feature_type == 'CNN': 
        #hist = get_hist_from_str(df["features_CNN"][index], separator = ',')
        x = np.arange(0, n_bins) 
        #plt.figure().set_figwidth(200)
        plt.bar(x, df["features_CNN"][index])
        plt.title("Feature Vector")
        plt.xlabel("Length")
        plt.ylabel("Value")
        plt.show()
        return 
    if feature_type == 'fusion': 
        #hist = get_hist_from_str(df["features_CNN"][index], separator = ',')
        x = np.arange(0, n_bins) 
        #plt.figure().set_figwidth(200)
        plt.bar(x, df["features_fusion"][index])
        plt.title("Feature Vector")
        plt.xlabel("Length")
        plt.ylabel("Value")
        plt.show()
        return 

def evaluate_all_features(df, image, measure = 'euclidean'):
    
    feature_types = ['histogram','CNN','fusion']
    for type in feature_types:
        print(f"----{type}----")
        df_fusion_query = query_image(df, image, measure=measure, feature_type = type)
        print("Match_query:", df_fusion_query['similarity'].head())
        print("Query Normalized Rank:",evaluate_query(df_fusion_query, image, measure=measure, normalized=True,feature_type= type)['similarity'].head())    

        print("----")

def precision_recall(df, matrix_, image_query, n_retrieved=10):
    # obtener la clase de la imagen
    image_class = SIFT.get_class(image_query)
    #print(image_class)
    # calcular similudes
    df_query_ = query_image(df, matrix_, image_query)
    #print(df_query)
    #print(df.columns)
    #N_rel = len(df_query[df_query["class"] == image_class])
    #print(N_rel)
    # get the top 10 images
    top_10 = df_query_.iloc[1:n_retrieved+1,:]
    # calculate TP
    TP = len(top_10[top_10["class"] == image_class])
    # calculate FP
    FP = len(top_10[top_10["class"] != image_class])
    # calculate FN
    FN = len(df_query_[df_query_["class"] == image_class]) - TP - 1

    #print(TP, FP, FN)
    # calculate precision
    precision_res = TP/(TP+FP)
    # calculate recall
    recall_res = TP/(TP+FN)
    return precision_res, recall_res

# create a similarity matrix for each image
def similarity_matrix(df, measure="cosine", feature_type = "histogram"):
    # create a matrix of zeros
    similarity_matrix = np.zeros((len(df), len(df)))
    # iterate over each image
    for i in tqdm.tqdm(range(len(df))):
        # get the image name
        image_name = df["image_name"][i]
        # calculate the similarity with the rest of images
        for j in range(len(df)):
            # get the image name
            image_name_2 = df["image_name"][j]
            if similarity_matrix[j,i] != 0:
                similarity_matrix[i, j] = similarity_matrix[j,i]
                continue
            if image_name == image_name_2:
                similarity_matrix[i, j] = 0
                continue
            # calculate the similarity
            similarity_matrix[i, j] = similarity_metric(df[f"features_{feature_type}"][i], df[f"features_{feature_type}"][j], measure=measure)
    return similarity_matrix

def consult_all(df_,image_example,measure, path_to_images):
    start = time.time()
    df_query_hand = query_image_normal(df_, image_example, measure=measure, feature_type = "histogram")
    print(f"Handcrafted query time:{np.round(time.time() - start,3)}")
    start = time.time()

    df_query_CNN = query_image_normal(df_,image_example, measure=measure, feature_type = "CNN")
    print(f"CNN query time:{np.round(time.time() - start,3)}")
    start = time.time()
    df_fusion_query = query_image_normal(df_, image_example, measure=measure, feature_type = "fusion")
    print(f"Fusion query time:{np.round(time.time() - start,3)}")
    # Create a figure with three rows and ten columns
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(15, 4))

    # Iterate over each row and column to plot the images
    for i in range(3):
        for j in range(10):
            if i == 0:
                axes[i, j].imshow(Image.open(f'{path_to_images}/{df_query_hand["image_name"][j]}'))
            elif i == 1:
                axes[i, j].imshow(Image.open(f'{path_to_images}/{df_query_CNN["image_name"][j]}'))
            else:
                axes[i, j].imshow(Image.open(f'{path_to_images}/{df_fusion_query["image_name"][j]}'))
            axes[i, j].axis('off')

    # Set the title for each row
    axes[0, 0].set_title('Handcrafted Features')
    axes[1, 0].set_title('CNN Features')
    axes[2, 0].set_title('Fused Features')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return