import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
df = pd.read_csv('nopunctANDstopwords.csv')

df[['title','text']] = df[['title','text']].replace(',',' ',regex=True)

def lemmatize(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmas)
    else:
        return text

df[['title','text']] = df[['title','text']].applymap(lemmatize)

df.to_csv('lemmatized.csv', index=False)


