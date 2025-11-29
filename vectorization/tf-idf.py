import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# load clean data; contains columns title_clean, text_clean, label
df = pd.read_csv('data/lemmatized.csv')

vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
)

# fit and transform on the cleaned text
tfidf_matrix = vectorizer.fit_transform(df['text_clean'])

# put TF–IDF matrix into a DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# combine TF–IDF features with the label
df_tfidf = pd.concat(
    [
        tfidf_df.reset_index(drop=True),
        df[['label']].reset_index(drop=True)
    ],
    axis=1
)

# save for modeling
df_tfidf.to_csv('data/tfidf_dataset.csv', index=False)

df_tfidf.head()
