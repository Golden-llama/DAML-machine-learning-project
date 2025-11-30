import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score



df = pd.read_csv('lemmatized.csv')

df['combined'] = df['title'] + ' ' + df['text']


tagged_docs = [
    TaggedDocument(words=row.split(), tags=[i])
    for i, row in enumerate(df["combined"])
]
model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)

model.build_vocab(tagged_docs)

model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

model.save("doc2vec_fakenews.model")

X = [model.dv[i] for i in range(len(tagged_docs))]
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

