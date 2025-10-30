import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv('nopunct.csv')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if isinstance(text, str):
        words = word_tokenize(text)
        filtered = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered)
    return text
df['text'] = df['text_no_punct'].apply(remove_stopwords)

df['title'] = df['title_no_punct'].apply(remove_stopwords)

