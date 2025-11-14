import pandas as pd
import string
import re

df = pd.read_csv('data/WELFake_Dataset.csv')

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


#saving the cleaned data into a folder called 'data'
df.to_csv("data/nopunct.csv", index =False)

print(df.text)
print(df.text_no_punct)





