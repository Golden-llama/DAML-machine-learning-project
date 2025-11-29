import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

df = pd.read_csv('data/lemmatized.csv')
df['title_clean'] = df['title_clean'].fillna('')
df['text_clean'] = df['text_clean'].fillna('')

tfidf_df = pd.read_csv('data/tfidf_dataset.csv')
tfidf_weights = tfidf_df.drop(columns=['label'])

w2v_model = Word2Vec.load('model/word2vec.model')

dim = w2v_model.vector_size

tokenized_texts = [word_tokenize(str(t)) for t in df['text_clean']]

def get_sentence_vector(words, model, tfidf_row, dim):
    vecs = []
    weights = []
    
    for word in words:
        if word in model.wv.key_to_index and word in tfidf_row.index:
            vecs.append(model.wv[word] * tfidf_row[word])
            weights.append(tfidf_row[word])
    
    if len(vecs) == 0:
        return np.zeros(dim)
    
    return np.average(vecs, axis=0, weights=weights)

vectors = np.vstack([
    get_sentence_vector(words, w2v_model, tfidf_df.iloc[i], dim)
    for i, words in enumerate(tokenized_texts)
])

columns = [f's2v_{i}' for i in range(dim)]
s2v_df = pd.DataFrame(vectors, columns=columns)
df_s2v = pd.concat([df.reset_index(drop=True), s2v_df.reset_index(drop=True)], axis=1)

df_s2v.to_csv('data/sentence2vec_dataset.csv', index=False)
df_s2v.head()

