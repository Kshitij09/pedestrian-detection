import pandas as pd
import os
from parser import VocParser
from dataclasses import asdict
from copy import copy
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import re
from functools import partial
from typing import List, Iterable
from functools import reduce
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import argparse

parser = argparse.ArgumentParser(description='Create annotations csv')
parser.add_argument('-d',dest='target_dir', type=str,
    help='Annotation files directory',
    default='PennFudanPed/Annotation')
parser.add_argument('-n',dest='fname', type=str,
    help="Output filename",
    default='data')

parser.add_argument('--seed',dest='seed', type=int,
    help="Random state for multilabel k-fold",
    default=47)


def f_standing(x: str): return x.endswith('Standing')
def f_walking(x: str): return x.endswith('Walking')

# from https://stackoverflow.com/a/44351664
def ilen(iterable: Iterable) -> int:
    return reduce(lambda sum, element: sum + 1, iterable, 0)

def get_count_string(label_list: List[str]) -> str:
    walk_count = ilen(filter(f_walking, label_list))
    stand_count = ilen(filter(f_standing, label_list))
    count_string = f"W{walk_count}S{stand_count}"
    return count_string

def create_folds_data(source_fname: str):
    """Create 'folds_data.csv' by extracting labels 
       for multilabel stratified split
    """
    marking = pd.read_csv(f"{source_fname}.csv")

    # Split bbox coordinates 
    marking[['x','y','x1','y1']] = pd.DataFrame(
        np.stack(marking['box'].apply(lambda x: ast.literal_eval(x))).astype(np.float32)
    )
    marking.drop(['box'],axis=1,inplace=True)

    # Calculate Area
    marking['area'] = (marking['x1']-marking['x']) * (marking['y1'] - marking['y'])

    # 'Box count': First label for the split
    df_folds = marking[['id']].copy()
    df_folds.loc[:,'box_count'] = 1
    df_folds = df_folds.groupby('id').count()

    # Media of 'Area': Second parameter for the split
    df_folds.loc[:,'area'] = marking[['id','area']]\
        .groupby('id')\
        .median()['area']

    # Label count string: Third parameter for the split
    # Eg. ["PASpersonWalking" "PASpersonWalking" "PASpersonStanding"] -> 'W2S1'
    df = marking.groupby('id')['label']\
        .apply(list)\
        .apply(get_count_string)\
        .reset_index(name='labels_agg')

    df_folds = pd.merge(df_folds,df,on='id')
    df_folds.to_csv(f'folds_{source_fname}.csv', index=False)

def extract_id(string: str,pat: re.Pattern) -> str:
    target = Path(string).name
    tokens = pat.search(target).groups()
    id = '_'.join(tokens)
    return id

def create_csv(target_dir: str, output: str):
    "Create 'data.csv' by parsing the annotation files"
    data = []
    ann_files = os.listdir(target_dir)
    for fname in ann_files:
        filepath = os.path.join(target_dir,fname)
        with open(filepath,'r') as f:
            raw_text = f.read()
            result = VocParser.parse(raw_text)
            for box in result.boxes:
                placeholder = asdict(result)
                del placeholder['boxes']
                placeholder['label'] = box.label
                placeholder['box'] = box.coord
                data.append(placeholder)

    dataframe = pd.DataFrame(data)

    # Extract id from filename
    # Eg. FudanPed00001 -> F_00001
    # PennPed00059 -> P_00059
    pat = re.compile(r'(\w)\D+(\d+).png')
    extract_fid = partial(extract_id,pat=pat)

    dataframe['id'] = dataframe['filename'].apply(extract_fid)
    dataframe = dataframe.reindex(columns=['id','filename','label','box'])
    dataframe.to_csv(f'{output}.csv', index=False)

def perform_split(source_fname: str):
    """Create 'data_with_folds.csv' by performing
       MultilabelStratifiedKFold on the target data
    """
    marking = pd.read_csv(f"{source_fname}.csv")
    df_folds = pd.read_csv(f"folds_{source_fname}.csv")
    
    image_ids = df_folds.id
    labels = df_folds.drop('id', axis=1).values
    mssf = MultilabelStratifiedKFold(n_splits=5,random_state=47, shuffle=True)
    df_folds.loc[:,'fold'] = -1

    for idx, (_,val_idx) in enumerate(mssf.split(X=image_ids,y=labels)):
        val_ids = image_ids[val_idx]
        df_folds.loc[df_folds['id'].isin(val_ids),'fold'] = idx+1

    final_df = pd.merge(marking,df_folds[['id','fold']],on='id')
    final_df.to_csv(f'{source_fname}_with_folds.csv', index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    create_csv(args.target_dir,args.fname)
    create_folds_data(args.fname)
    perform_split(args.fname)