#import numpy as np
#aqui posibles utils en el futuro que sean demasiado grandes pa poner 
#en el notebook comodamente


def load_and_find_padding(paths = ['data/GPR1200/images/'], valid_types = [".jpg",".JPEG"]):
    """ load and process saves memory by only saving the vectorized image and not the image itself"""
    hists = []

    for i in range(len(paths)):
        list_dir = notebook.tqdm(os.listdir(paths[i]))
        
        for f in list_dir:
            list_dir.set_description(f"Loading images from directory {paths[i]}")
            ext = os.path.splitext(f)[1]
            
            if ext.lower() not in valid_types:
                continue
            image = (Image.open(os.path.join(paths[i],f)).convert('L'))
            hist = classic_method(image)
            hists.append(hist)
    return hists