import pandas as pd

# check for nopunc + stopwords ------------------------------------
df = pd.read_csv("data/nopunct_stopwords.csv")

# print all column names
print(df.columns)
print() # newline
print("ORIGINAL TEXT:")
print(df['text'].iloc[0][:300]) # print first 300 chars
print()
print("CLEANED TEXT (after punct + stopwords):")
print(df['text_no_punct'].iloc[0][:300])



# check for lemmatization ------------------------------------
df = pd.read_csv("data/lemmatized.csv")

# print all column names
print(df.columns)
print() # newline
print("CLEANED TEXT (after lemmatization):")
print(df['text_clean'].iloc[0][:300])