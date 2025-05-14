📰 Fake News Detector

A full-stack NLP + ML project that detects **FAKE** or **REAL** news articles using Natural Language Processing and Machine Learning techniques. 
The project is written in Python and uses the Naive Bayes algorithm with TF-IDF vectorization for classification. It includes full preprocessing, training, evaluation,
model saving, and deployment.
___________________________________________________________________________________________________________________________________________________________________________________
⚙️ Stack Used

- **Pandas / NumPy** – data manipulation  
- **NLTK** – natural language preprocessing  
- **Scikit-learn** – model training and evaluation  
- **Joblib** – model serialization  
- **Python** – scripting and execution  
- **Streamlit** – deployment
___________________________________________________________________________________________________________________________________________________________________________________
📂 Project Structure

fake-news-detector/
│
├── news.csv # Dataset (text + label)
├── fake_news_detector.py # Core pipeline (preprocessing, training, testing)
├── logistic_fake_news.pkl # Trained model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── app/ # Web interface (Streamlit)
│ └── app.py
├── README.md # Project documentation

_______________________________________________________________________________________________________________________________________________________________________________
📁 Dataset
The dataset should be a CSV file with the following structure:

| text                                | label |
|-------------------------------------|-------|
| "The president made a new law..."   | REAL  |
| "Aliens found under the Pentagon"   | FAKE  |

You can download a dataset from [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
_______________________________________________________________________________________________________________________________________________________________________________
🚀 Run Locally

### Development Mode

# Clone the repository
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the training & testing script
python fake_news_detector.py

# Navigate to the app folder
cd app

# If using Streamlit
streamlit run app.py

# If using Flask
python app.py
________________________________________________________________________________________________________________________________________________________________________________
 Running Predictions
 -------------------

#Inside fake_news_detector.py, you can run:
test_custom_news("The president held a press conference on healthcare.")
test_custom_news("NASA confirms that aliens are disguised as humans.")

Output:
Prediction: REAL
Prediction: FAKE
_______________________________________________________________________________________________________________________________________________________________________________
Model Info
----------
Vectorizer: TfidfVectorizer (max_features=5000)

Classifier: Multinomial Naive Bayes

Evaluation Metrics: Accuracy, Precision

Accuracy: ~91%

Precision: ~89%
_______________________________________________________________________________________________________________________________________________________________________________
