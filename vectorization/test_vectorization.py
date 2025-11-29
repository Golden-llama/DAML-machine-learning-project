import pandas as pd

w2v = pd.read_csv("data/word2vec_dataset.csv")
w2v.shape
w2v['label'].value_counts()