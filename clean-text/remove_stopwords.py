import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv('data/nopunct.csv')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if isinstance(text, str):
        words = word_tokenize(text)
        filtered = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered)
    return text

df['text_no_punct'] = df['text_no_punct'].apply(remove_stopwords)
df['title_no_punct'] = df['title_no_punct'].apply(remove_stopwords)

# save output into data folder
df.to_csv('data/nopunct_stopwords.csv', index=False)

