import pandas as pd 
import numpy as np
from tqdm import tqdm

def generate_hists(csv_path:str,output_path:str, n_clusters:int):
    print(f'Generating histograms using total_words_quant @ {csv_path}')
    #group by image_name and calculate the histogram of visual words
    for_quatinzation = pd.read_csv(f'{csv_path}/total_words_quant.csv')
    for_quatinzation = for_quatinzation.drop(columns = ['index', 'class'])
    #print(for_quatinzation.iloc[0])
    for_quatinzation = for_quatinzation.groupby('image_name').agg(lambda x: x.tolist())
    for_quatinzation = for_quatinzation.reset_index()
    for_quatinzation = for_quatinzation[:-1]
    for_quatinzation['histogram'] = for_quatinzation['visual_word'].apply(lambda x: np.histogram(np.array(x).astype(float), bins = n_clusters, density=True)[0])
    #print(for_quatinzation['visual_word'])
    for_quatinzation = for_quatinzation.drop(columns = ['visual_word'])
    for_quatinzation.to_csv(f'{output_path}/total_quant_histogram.csv', index = False)
    print(f'Output has been saved as total_quant_histogram.csv @ {output_path}')
 