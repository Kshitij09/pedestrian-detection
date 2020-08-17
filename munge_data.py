import pandas as pd
import os
from parser import VocParser
from dataclasses import asdict
from copy import copy
import argparse

parser = argparse.ArgumentParser(description='Create annotations csv')
parser.add_argument('-d',dest='target_dir', type=str,
    help='Annotation files directory',
    default='PennFudanPed/Annotation')
parser.add_argument('-n',dest='fname', type=str,
    help="Output filename",
    default='data')

def create_csv(target_dir: str, output: str):
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
    dataframe.to_csv(f'{output}.csv')

if __name__ == "__main__":
    args = parser.parse_args()
    create_csv(args.target_dir,args.fname)
