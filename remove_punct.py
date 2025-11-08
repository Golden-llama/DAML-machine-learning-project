import pandas as pd
import string
import re

df = pd.read_csv('WELFake_Dataset.csv')

print(df.columns)

#removing punctuations

def remove_punctuation(text):
    if isinstance(text, str):
        text = re.sub(r"[“”‘’–—…]", "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return re.sub(r'\s+', ' ', text).strip()
    
  
    return text

df["text_no_punct"] = df["text"].apply(remove_punctuation)
df["title_no_punct"] = df["title"].apply(remove_punctuation)

df.to_csv("nopunct.csv", index =False)

print(df.text)
print(df.text_no_punct)





