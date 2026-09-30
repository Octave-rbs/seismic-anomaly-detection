import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"dataset")
FILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"dataset/test")

def load_annotations(path: str) -> dict:
    """
    Load the annotations: {filename: {"horizontal": bool, "other": bool}}.
    """
    with open(path, "r") as f:
        return json.load(f)

def load_patch(path: str) -> np.ndarray:
    """
    Load a patch and standardize it (zero mean, unit variance).
    """
    patch = np.load(path).astype(np.float32).squeeze()
    return (patch - patch.mean()) / (patch.std() + 1e-8)

def compare_method_to_annotations(method = lambda x : {'horizontal':False,'other':False}, show = False):

    ANNOTATIONS = os.path.join(DATASET_DIR, "annotations.json")
    annotations = load_annotations(ANNOTATIONS)
    results = {'horizontal_false_+':[0,[]],'horizontal_false_-':[0,[]],'other_false_+':[0,[]],'other_false_-':[0,[]]}

    for file, annot in tqdm(annotations.items(), desc="Processing annotations"):
        path = os.path.join(FILE_DIR, file)
        pred = method(path)

        if pred['horizontal'] and not(annot['horizontal']):
            results['horizontal_false_+'][0] += 1
            results['horizontal_false_+'][1].append(path)
        if not(pred['horizontal']) and annot['horizontal']:
            results['horizontal_false_-'][0] += 1
            results['horizontal_false_-'][1].append(path)
        if pred['other'] and not(annot['other']):
            results['other_false_+'][0] += 1
            results['other_false_+'][1].append(path)
        if not(pred['other']) and annot['other']:
            results['other_false_-'][0] += 1
            results['other_false_-'][1].append(path)

    for key,count in results.items():
        print(key+' : '+str(count[0])+'/'+str(len(annotations.keys())))
        if count[0]>0 and show:
            index = int(np.random.rand()*count[0])
            path = results[key][1][index]
            plt.title(key+' '+str(annotations[os.path.basename(path)]))
            plt.imshow(load_patch(path), origin="lower")
            plt.show()
    horiz_acc =  1 - (results['horizontal_false_+'][0] + results['horizontal_false_-'][0])/len(annotations.keys())
    print('Horizontal accuracy : ', horiz_acc)
    other_acc = 1 - (results['other_false_+'][0] + results['other_false_-'][0])/len(annotations.keys())
    print('Other accuracy : ', other_acc)


compare_method_to_annotations()
    