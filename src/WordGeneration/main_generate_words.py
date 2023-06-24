import joblib
import pandas as pd
import tqdm
import sys


"""def generate_words(csv_file: str, output_path:str, model):
    print(f'Generating words using images_descriptors_quant.csv @ {csv_file}')
    # Get the total number of lines in the CSV file 
    csv_file = f'{csv_file}/images_descriptors_quant.csv'
    total_lines = sum(1 for line in open(csv_file, 'r'))
    # Create a tqdm progress bar and set the total number of iterations
 
    for_quatinzation = pd.read_csv(csv_file)
        

    inferece_model = joblib.load(model)

    image_name = None
    index = 0

    
    for index, row in tqdm.tqdm(for_quatinzation.iterrows(), total=len(for_quatinzation), desc='Processing'):
        img_name_aux = row['image_name']
        if img_name_aux != image_name:
            image_name = img_name_aux
            index = 0
        for_quatinzation.loc[index, 'index'] = index
        index += 1

    # Predict on quantization dataset
    for_quatinzation['visual_word'] = inferece_model.predict(for_quatinzation.drop(columns=['image_name', 'class', 'index']))
    # Save quantization dataset dropping the descriptors
    for_quatinzation.drop(columns=[str(i) for i in range(128)], inplace=True)
    for_quatinzation.to_csv(f'{output_path}/total_words_quant.csv', index=False)
    print(f'Output has been saved as total_words_quant.csv @ {output_path}')
"""

 
def generate_words(csv_file: str, output_path: str, model):
    print(f'Generating words using images_descriptors_quant.csv @ {csv_file}')
    csv_file = f'{csv_file}/images_descriptors_quant.csv'
    total_lines = sum(1 for _ in open(csv_file, 'r'))

    inferece_model = joblib.load(model)
    image_name = None
    index = 0

    with tqdm.tqdm(total=total_lines, desc='Generating words', file=sys.stdout) as pbar:
        for chunk in pd.read_csv(csv_file, chunksize=1000):
            chunk = chunk.fillna(0)
            for index, row in chunk.iterrows():
                img_name_aux = row['image_name']
                if img_name_aux != image_name:
                    image_name = img_name_aux
                    index = 0
                chunk.loc[index, 'index'] = index
                index += 1

            chunk['visual_word'] = inferece_model.predict(chunk.fillna(0).drop(columns=['image_name', 'class', 'index']))
            chunk.drop(columns=[str(i) for i in range(128)], inplace=True)
            chunk.to_csv(f'{output_path}/total_words_quant.csv', mode='a', index=False)
            pbar.update(len(chunk))

    print(f'Output has been saved as total_words_quant.csv @ {output_path}')
