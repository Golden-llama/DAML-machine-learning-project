import pandas as pd

df = pd.read_csv("data/nopunct_stopwords.csv")

# print all column names
print(df.columns)
print() # newline
print("ORIGINAL TEXT:")
print(df['text'].iloc[0][:300]) # print first 300 chars
print()
print("CLEANED TEXT (after punct + stopwords):")
print(df['text_no_punct'].iloc[0][:300])