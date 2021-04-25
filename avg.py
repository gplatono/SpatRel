import pandas as pd
import numpy as np

with open('StrctureData.csv', 'r') as csvfile:
    df = pd.read_csv(csvfile, delimiter=',', quotechar='"')
    df = df.groupby(by= ['h1','h2']).mean()
    for col in df.columns.tolist():
	    print(col + ":", df[col].idxmax())
    df.to_csv('avg.csv', index_label = ['h1','h2'])