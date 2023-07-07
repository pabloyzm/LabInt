import cv2
import os
import tqdm
import pandas as pd  
from sklearn.model_selection import train_test_split

class SIFTFeatures:
    """Will return a csv of flattened descriptors. Receives a directory that contains images, ndescriptors, subsample ratio"""

    def __init__(self,SIFT_input_path, output_path,n_features = 300, subsample_ratio =1.0, run = True) -> None:
        self.SIFT_output_path = output_path
        self.filename = 'images_descriptors'
        self.n_features = n_features
        self.subsample_ratio = subsample_ratio
        if run:
            self.descriptors_list = self.load_and_process(self.SIFT_input_path)

    def load_and_process(self, input_path,output_path, valid_types=['.jpg','.JPG','.jpeg' ,'.JPEG']):
        """load and process saves memory by only saving the descriptors and not the image itself
        """
        descriptors_for_kmeans_list = []
        descriptors_for_quantization_list = []

        list_dir = os.listdir(input_path)
        # make a dictionary with the image name and the class
        list_dir = [f for f in list_dir if os.path.splitext(f)[1].lower() in valid_types]
        labels = [self.get_class(f) for f in list_dir]
        if self.subsample_ratio != 1:
            list_dir_for_kmeans, list_dir_for_quantization, labels_for_kmeans, labels_for_quantization = train_test_split(list_dir, labels, train_size=self.subsample_ratio, stratify=labels,  random_state=1)
        else:
            list_dir_for_kmeans, list_dir_for_quantization = list_dir, list_dir

        # print('Creating a csv file of descriptors from a subset of images in the directory...')
        # for f in tqdm.tqdm(list_dir_for_kmeans):
        #     ext = os.path.splitext(f)[1]
        #     if ext.lower() not in valid_types:
        #         continue
        #     image = cv2.imread(os.path.join(input_path, f))
        #     des_kmeans = self.generate_descriptor(image)  # Assuming generate_descriptor function is defined elsewhere
        #     # generate a dataframe with the descriptors and a column with the image name
        #     df_aux = pd.DataFrame(des_kmeans)
        #     df_aux['image_name'] = f
        #     df_aux['class'] = self.get_class(f)
        #     descriptors_for_kmeans_list.append(df_aux)
        # descriptors_for_kmeans_list = pd.concat(descriptors_for_kmeans_list)
        # self.to_csv(descriptors_for_kmeans_list,subindex='kmeans')
        # del descriptors_for_kmeans_list

        if self.subsample_ratio < 1.0:
            print('Creating a csv file of descriptors for quantization...')
            for f in tqdm.tqdm(list_dir_for_quantization):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_types:
                    continue
                image = cv2.imread(os.path.join(input_path, f))
                des_quant = self.generate_descriptor(image)  # Assuming generate_descriptor function is defined elsewhere
                # generate a dataframe with the descriptors and a column with the image name
                df_aux = pd.DataFrame(des_quant)
                df_aux['image_name'] = f
                df_aux['class'] = self.get_class(f)
                descriptors_for_quantization_list.append(df_aux)
            descriptors_for_quantization_list = pd.concat(descriptors_for_quantization_list)
            self.to_csv(descriptors_for_quantization_list,subindex='quant')
        return 0

    def get_class(self, img_path):
        img_path = img_path.lower()
        img_path = img_path.replace(".jpg", "").replace(".jpeg", "")
        if len(img_path.split("_")) > 1:
            return img_path.split("_")[0] + "_" + "type1"
        else:
            class_name = str((int(img_path) - 100000) // 100) + "_" + "type2"
            return class_name

    def generate_descriptor(self,image):
        # Convert the image to grayscale (SIFT works on grayscale images)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Initiate SIFT detector
        sift = cv2.SIFT_create(nfeatures=self.n_features)
        # Find keypoints and dfcriptors directly
        kp, des = sift.detectAndCompute(gray, None)
        # del kp
        # des = np.array(des).flatten()
        # print(kp)
        return des 
    def to_csv(self,df,subindex = ''):

        try:
            assert type(self.filename) == str
            print("Saving df as csv file...", end="\r")
            df.to_csv(f'{self.SIFT_output_path}/{self.filename}'+'_'+subindex+'.csv',index = False)
            print("Finished saving.")
            del df
        except AssertionError as a:
            print(a)
        return 0

##runtime 

 