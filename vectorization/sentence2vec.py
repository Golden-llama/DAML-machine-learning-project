import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('punkt')

# load lemmatized dataset
df = pd.read_csv('data/lemmatized.csv')
df['text_clean'] = df['text_clean'].fillna('')

texts = df['text_clean'].tolist()

def nltk_tokenizer(text):
    return word_tokenize(text.lower())

vectorizer = TfidfVectorizer(
    tokenizer=nltk_tokenizer,
    lowercase=False,
    ngram_range=(1, 1),   # unigrams only
    max_features=5000,
    sublinear_tf=True     # log(1 + tf) for smoother weights
)

tfidf_matrix = vectorizer.fit_transform(texts)
vocab = vectorizer.vocabulary_

w2v_model = Word2Vec.load('model/word2vec.model')
dim = w2v_model.vector_size

tokenized_texts = [nltk_tokenizer(t) for t in texts]

def get_weighted_doc_vector(doc_idx, tokens, model, tfidf_matrix, vocab, dim):
    row = tfidf_matrix[doc_idx]
    indices = row.indices
    data = row.data
    weight_lookup = {idx: val for idx, val in zip(indices, data)}

    vecs = []
    weights = []

    for tok in tokens:
        if tok in model.wv.key_to_index:
            col_idx = vocab.get(tok)
            if col_idx is not None:
                w = weight_lookup.get(col_idx, 0.0)
                if w > 0.0:
                    # normalize word vector before weighting
                    vec = model.wv[tok]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vecs.append(vec * w)
                    weights.append(w)

    if not vecs:
        return np.zeros(dim)

    weight_sum = sum(weights)
    if weight_sum == 0:
        return np.mean(vecs, axis=0)

    return np.average(vecs, axis=0, weights=weights)

vectors = np.vstack([
    get_weighted_doc_vector(i, tokenized_texts[i], w2v_model, tfidf_matrix, vocab, dim)
    for i in range(len(df))
])

# save as s2v_0 ... s2v_99 + label
columns = [f's2v_{i}' for i in range(dim)]
s2v_df = pd.DataFrame(vectors, columns=columns)

df_s2v = pd.concat(
    [
        s2v_df.reset_index(drop=True),
        df[['label']].reset_index(drop=True)
    ],
    axis=1
)

df_s2v.to_csv('data/sentence2vec_dataset.csv', index=False)
print(df_s2v.head())
print("Saved Sentence2Vec features to data/sentence2vec_dataset.csv")
