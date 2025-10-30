import pandas as pd
import string
import re

df = pd.read_csv('WELFake_Dataset.csv')

print(df.columns)

#removing punctuations

def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

df["text_no_punct"] = df["text"].apply(remove_punctuation)
df["title_no_punct"] = df["title"].apply(remove_punctuation)

df.to_csv("nopunct.csv", index =False)

#print(df.text_no_punct)
#print(df.text)






