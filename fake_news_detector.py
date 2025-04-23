# fake_news_detector.py

import pandas as pd
import numpy as np
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load your dataset (assumes a 'text' and 'label' column)
df = pd.read_csv("news.csv")  # Replace with your actual CSV path

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(clean_text)

# Convert labels
X = df['cleaned_text']
y = df['label'].map({'REAL': 1, 'FAKE': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'logistic_fake_news.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Test custom input
def test_custom_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    print("\nInput:", text)
    print("Prediction:", label)

# Test samples
test_custom_news("The president held a press conference to announce new healthcare reforms.")
test_custom_news("NASA confirms that aliens are living among us disguised as humans.")
test_custom_news("Breaking: Scientists admit that the Earth is flat after all!")
