import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 



df = pd.DataFrame('lemmatized.csv')

df['document'] = df['title'] + ' ' + df['text']
def tag_and_tokenize(doc_series):
    
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
                   for i, _d in enumerate(doc_series)]
    return tagged_data


tagged_data = tag_and_tokenize(df['document'])
model = Doc2Vec(vector_size=100, # 100-dimensional vectors
                min_count=2,      # Ignore words that appear less than 2 times
                epochs=40)        # Number of iterations over the dataset

model.build_vocab(tagged_data)

model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

doc_vectors = []

for i in range(len(tagged_data)):
    
    vector = model.dv[str(i)]
    doc_vectors.append(vector)

X = pd.DataFrame(doc_vectors)
y = df['label']