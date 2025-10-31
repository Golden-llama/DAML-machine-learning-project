import pandas as pd
import string
import re

df = pd.read_csv('nostopwords.csv')

print(df.columns)

#removing punctuations

def remove_punctuation(text):
    if isinstance(text, str):
        text = re.sub(r"[“”‘’–—…]", "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return re.sub(r'\s+', ' ', text).strip()
    
  
    return text

df["text"] = df["text"].apply(remove_punctuation)
df["title"] = df["title"].apply(remove_punctuation)

df.to_csv("nopunctANDstopwords.csv", index =False)

#print(df.text_no_punct)
#print(df.text)






