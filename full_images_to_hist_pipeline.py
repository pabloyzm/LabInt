from src.SIFT.SIFT_gen_and_utils import SIFTFeatures
from src.Clusterization.generate_cluster_model import initialize_kmeans
from src.WordGeneration.main_generate_words import generate_words
from src.HistogramGeneration.main_generate_histograms import generate_hists
import yaml 
import joblib
import numpy as np
import os 

class collection_to_hists(SIFTFeatures):
    def __init__(self, config_file:str, ) -> None:
        self.config_file = config_file
        #INITIAL CONFIG
        col_size = os.get_terminal_size().columns
        self.setup = self.load_yaml() 

        if self.setup['execution_pipeline']['SIFT_descriptor_gen']:
            print(col_size*'-')
            self.filename = 'images_descriptors' 
            self.SIFT_input_path = self.setup['SIFT_config']['input_path']
            self.SIFT_output_path = self.setup['SIFT_config']['output_path']
            self.n_features = self.setup['SIFT_config']['number_of_descriptors']
            self.subsample_ratio = self.setup['SIFT_config']['subsample_ratio']
            #SIFT GENERATION
            self.descriptors_list = self.load_and_process(self.SIFT_input_path,self.SIFT_output_path)
        print(col_size*'-')

       
        #CLUSTERING
        self.model_save_path = self.setup['clusterization_config']['save_model_path']
        self.n_clusters = self.setup['clusterization_config']['n_clusters']
        if self.setup['execution_pipeline']['clusterization']:
            self.clusterization_input_csv = self.setup['clusterization_config']['input_path']
            self.elbow = self.setup['clusterization_config']['perform_elbow_method_for_tuning']
            if self.elbow:
                self.elbow_vector = self.setup['clusterization_config']['elbow_method_vector']
                print(f'Using elbow vector {self.elbow_vector}')
            self.plot_clustering = self.setup['clusterization_config']['kmeans_plot']
            self.kmeans_model, self.n_clusters = initialize_kmeans(self.clusterization_input_csv,
                                                                self.elbow,
                                                                np.linspace(*self.elbow_vector, dtype = int),
                                                                self.plot_clustering)
            joblib.dump(self.kmeans_model, f"{self.model_save_path}/kmeans_{self.n_clusters}.pkl")
            del self.kmeans_model 
        print(col_size*'-')

         
        #WORD GENERATION
        if self.setup['execution_pipeline']['word_gen']:
            self.load_model = f'{self.model_save_path}//kmeans_{self.n_clusters}.pkl'
            self.word_input_path = self.setup['word_gen_config']['input_path']
            self.word_output_path = self.setup['word_gen_config']['output_path']
            generate_words(self.word_input_path,self.word_output_path,self.load_model)
        print(col_size*'-')


        #HISTOGRAM GENERATION
        if self.setup['execution_pipeline']['hist_gen']:
            self.hist_gen_input_path = self.setup['histogram_gen_config']['input_path']
            self.hist_csv_ouput_path = self.setup['histogram_gen_config']['output_path']
            generate_hists(self.hist_gen_input_path,self.hist_csv_ouput_path,self.n_clusters)
        print(col_size*'-')


    def load_yaml(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config


collection_to_hists('/root/labint/scripts/LabInt/config/full_images_to_hist_config.yaml')