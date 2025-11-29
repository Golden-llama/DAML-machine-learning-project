import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

df = pd.read_csv('data/nopunct_stopwords.csv')

def lemmatize(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmas)
    return text

# cleaned columns
df['text_clean'] = df['text_no_punct'].apply(lemmatize)
df['title_clean'] = df['title_no_punct'].apply(lemmatize)

# keep cleaned title, text, and label
out = df[['title_clean', 'text_clean', 'label']]

# save data set
out.to_csv('data/lemmatized.csv', index=False)

