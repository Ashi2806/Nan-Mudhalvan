import pandas as pd
import string
import joblib
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
   text = text.lower()
   text = "".join([char for char in text if char not in string.punctuation])
   tokens = text.split()
   tokens = [word for word in tokens if word not in stop_words]
   return " ".join(tokens)

def load_and_preprocess(filepath):
  df = pd.read_csv("sample1.csv")
  if 'Review' not in df.columns or 'Sentiment' not in df.columns:
    raise ValueError("CSV must contain 'Review' and 'Sentiment' columns.")

  df.dropna(subset=['Review', 'Sentiment'], inplace=True)
  df["clean_review"] = df["Review"].apply(clean_text)
  return df
def train_and_evaluate(df):
  X = df["clean_review"]
  y = df["Sentiment"]
# Stratified split to maintain class balance
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

  pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(solver="liblinear"))
   ])

  pipeline.fit(X_train, y_train)
  
  y_pred = pipeline.predict(X_test)

  print("\nModel Evaluation:")
  print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))
  pipeline.fit(X, y)
  joblib.dump(pipeline, "sentiment_model.pkl")
  print("\n✅ Model trained and saved as 'sentiment_model.pkl'")
if __name__== "__main__":
    
    try:
     df = load_and_preprocess("sample1.csv")
     train_and_evaluate(df)
    except Exception as e:
      print(f"❌ Error: {e}")