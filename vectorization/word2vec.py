import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

df = pd.read_csv('data/lemmatized.csv')
df['text_clean'] = df['text_clean'].fillna('')

# tokenize text
tokenized_texts = [word_tokenize(str(t).lower()) for t in df['text_clean']]

# train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,   # dimensionality of embeddings
    window=5,
    min_count=2,
    workers=4,
    sg=1               # skip-gram
)

w2v_model.save('data/word2vec.model')

# function to get a document vector by averaging word vectors
def get_doc_vector(words, model, dim):
    # keep only words that appear in the model's vocab
    valid_words = [w for w in words if w in model.wv.key_to_index]
    if len(valid_words) == 0:
        return np.zeros(dim)
    return np.mean(model.wv[valid_words], axis=0)

dim = w2v_model.vector_size

# build document embeddings
vectors = np.vstack([get_doc_vector(words, w2v_model, dim) for words in tokenized_texts])

# build dataframe of embeddings
columns = [f'w2v_{i}' for i in range(dim)]
w2v_df = pd.DataFrame(vectors, columns=columns)

# attach label only
df_w2v = pd.concat(
    [
        w2v_df.reset_index(drop=True),
        df[['label']].reset_index(drop=True)
    ],
    axis=1
)

# save dataset for classifiers
df_w2v.to_csv('data/word2vec_dataset.csv', index=False)

print(df_w2v.head())
