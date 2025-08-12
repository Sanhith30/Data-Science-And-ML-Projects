# Fake News Detection — End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LogisticRegression-orange) 
![Streamlit](https://img.shields.io/badge/Deployed%20With-Streamlit-brightgreen) 
![Status](https://img.shields.io/badge/Status-Complete-success)  

Predict whether a news article is **FAKE** or **REAL** based on its textual content.  

This project demonstrates the **entire ML lifecycle** — from **data preprocessing, feature engineering, model building, and evaluation** to **deployment as a web application**.  
It covers **data cleaning, EDA, NLP-based feature extraction, model building, evaluation, and deployment**.

---

## 🛠 Tech Stack
**Languages & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit  
**ML Algorithm:** Logistic Regression (TF-IDF Vectorized text data)  
**Deployment:** Streamlit  

---

##  Project Structure
```
fake-news-detection/
│
├── fake_news.ipynb         # Notebook with EDA, preprocessing, model training
├── app.py                  # Streamlit web app
├── fake_news_model.pkl     # Saved ML model
├── vectorizer.pkl          # Saved TF-IDF vectorizer (not included due to size)
├── requirements.txt        # Dependencies
└── data/                   # Dataset files (from Kaggle)
```

---

## 📊 Dataset
**Source:** [Kaggle - Fake News Dataset](https://www.kaggle.com/c/fake-news/data)  

**Features:**
- **Title** — Headline of the article  
- **Text** — News article content  
- **Author** — Author of the article  
- **Subject** — Category of the article  

**Target:**
- **Label** — 1 = Fake, 0 = Real  

---

## 🛠 Workflow
1. **Data Preprocessing**  
   - Remove missing values  
   - Clean text data (lowercasing, removing punctuation, stopwords, etc.)  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize class distribution  
   - Analyze word frequency for fake vs real  

3. **Feature Engineering**  
   - Convert text to numerical vectors using **TF-IDF Vectorizer**  

4. **Model Training**  
   - Logistic Regression classifier  

5. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  

6. **Deployment**  
   - Interactive Streamlit app for live predictions  

---

## 📈 Results & Insights
- **Accuracy:** ~92% (Logistic Regression with TF-IDF)  
- **Key Findings:**  
  - Fake news articles often have more sensational words and clickbait titles  
  - Real news tends to have more structured sentences and factual reporting  

---

## ⚠ Missing `vectorizer.pkl`
Due to file size limitations, `vectorizer.pkl` is **not included**.  
**Alternative:** Recreate it by running `fake_news.ipynb`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.7)
vectorizer.fit(X_train)  # X_train from your dataset split

import pickle
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
```

---

## 💻 Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sanhith30/Data-Science-And-ML-Projects.git
cd "Fake News Detection"
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook (to train model & vectorizer)
```bash
jupyter notebook fake_news.ipynb
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---



---

## 📸 Screenshots
![Fake News App Screenshot](images/Screenshot 2025-08-12 113658.png)

---

## 🚀 Future Improvements
- Try advanced NLP models (BERT, RoBERTa) for better accuracy  
- Perform hyperparameter tuning  
- Add explainability (SHAP values)  
- Improve UI with news scraping feature  

---

## 🧠 Skills Demonstrated
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Natural Language Processing (NLP)  
- Machine Learning Model Training  
- Model Evaluation  
- Web App Deployment  

---

## 📬 Contact
**Author:** THIKKAVARAPU SANHITH  
[LinkedIn](https://linkedin.com/in/sanhith30) | [GitHub](https://github.com/sanhith30)  
