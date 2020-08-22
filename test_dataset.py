import pandas as pd
import numpy as np
import ast
from peddet.dataset import PennFudanDataset
if __name__ == "__main__":
    fold = 2
    df = pd.read_csv('data_with_folds.csv')
    df[['x','y','x1','y1']] = pd.DataFrame(
        np.stack(df['box'].apply(ast.literal_eval)).astype(np.float32)
    )    

    train_df = df.loc[df['fold'] != fold].copy()
    train_dataset = PennFudanDataset(train_df)
    print(train_dataset[0])