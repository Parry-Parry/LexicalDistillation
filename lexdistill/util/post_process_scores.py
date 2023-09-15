from fire import Fire
import numpy as np 
import pandas as pd 
import json 
from collections import defaultdict
def main(score_path : str):
    with open(score_path) as f:
        scores = json.load(f)
    
    new = defaultdict(lambda : defaultdict(lambda : 0.))
    for k, v in scores.items():
        for _k, _v in v.items():
            new[k][_k] = float(_v) if not np.isnan(float(_v)) else 0.
    
    with open(score_path, 'w') as f:
        json.dump(new, f)