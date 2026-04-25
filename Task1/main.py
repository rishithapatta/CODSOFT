import pandas as pd
import re
df = pd.read_csv("train_data.txt", sep=":::", engine="python")
df.columns = ["id", "title", "genre", "plot"]
df['genre'] = df['genre'].str.strip()
df['plot'] = df['plot'].str.strip()
df = df.sample(20000, random_state=42)
print(df.head())
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
df['clean_plot'] = df['plot'].apply(clean_text)
print(df[['plot', 'clean_plot']].head())
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['clean_plot'])
y = df['genre']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
# Custom input test
sample = ["A young boy discovers magical powers and fights evil forces"]
sample_clean = [clean_text(text) for text in sample]
sample_tfidf = tfidf.transform(sample_clean)
prediction = model.predict(sample_tfidf)
print("Predicted Genre:", prediction[0])
import pickle


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)