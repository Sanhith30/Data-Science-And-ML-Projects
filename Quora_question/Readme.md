# QUORA QUESTION PAIR SIMILARITY PROJECT NOTES

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question.  

Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.  

👉 The **main aim** of this project is predicting whether a pair of questions are similar or not.  
This could be useful to instantly provide answers to questions that have already been answered.  

**Dataset Credits:** Kaggle  

---

## 📌 Problem Statement
Identify which questions asked on Quora are duplicates of questions that have already been asked.

---

## 🎯 Business Objectives & Constraints
- The cost of a mis-classification can be very high.  
- Need probability outputs so threshold can be chosen.  
- No strict latency concerns.  
- Interpretability is partially important.  

**Performance Metric:**
- Log-loss  
- Binary Confusion Matrix  

---

## 📂 Project Structure
This repository contains the following notebooks:

1. **[1.Quora_Questions(Basic).ipynb](./1.Quora_Questions(Basic).ipynb)** – Basic dataset exploration and understanding.  
2. **[2.Quore_Que_preprocessing(advanced_FE).ipynb](./2.Quore_Que_preprocessing(advanced_FE).ipynb)** – Data preprocessing and advanced feature engineering.  
3. **[3.Mean_tfidf_W2V.ipynb](./3.Mean_tfidf_W2V.ipynb)** – Vectorization using TF-IDF and TF-IDF Word2Vec.  
4. **[4.Model_Training.ipynb](./4.Model_Training.ipynb)** – Model training, evaluation, and performance comparison.  

---

## 📊 BASICS
- `train.csv` has the data  
- Features → `id (int)`, `qid1 (int)`, `qid2 (int)`, `question1 (string)`, `question2 (string)`  
- Label → `is_duplicate (int)`  
- Binary classification problem  
- Metrics → log loss & binary confusion matrix  

---

## 📊 EDA
- Bar chart of duplicate vs non-duplicate pairs  
- Bar chart of repeated vs non-repeated questions  
- Checking duplicate data points  
- Log histogram of frequency of questions  
- Null value check → filled with empty strings  

---

## 🛠️ Feature Extraction
- freq_qid1, freq_qid2  
- q1len, q2len, q1_n_words, q2_n_words  
- word_common, word_total, word_share  
- freq_q1+freq_q2, freq_q1-freq_q2  
- Avg. word share & common words higher in duplicate pairs  
- Distribution of word_common overlaps in similar/non-similar pairs  

---

## 🧹 Preprocessing of Text
- Remove HTML tags  
- Remove punctuations  
- Perform stemming  
- Remove stopwords  
- Expand contractions (e.g., *wasn't → was not*)  

---

## ⚙️ Advanced Feature Extraction
**Definitions**  
- Token = split sentence by space  
- Stop_Word = stopwords from NLTK  
- Word = token not a stopword  

**Features**  
- cwc_min, cwc_max → common words ratio  
- csc_min, csc_max → common stopwords ratio  
- ctc_min, ctc_max → common tokens ratio  
- last_word_eq, first_word_eq  
- abs_len_diff, mean_len  
- fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio  
- longest_substr_ratio  

---

## 📑 Analysis of Extracted Features
- Word cloud of duplicate pairs  
- Word cloud of non-duplicate pairs  
- 15+ NLP features created  

---

## 📈 Visualization
- t-SNE & PCA → 15D data → 3D  

---

## 📐 Vectorization
- TF-IDF dictionary (word → tf-idf score)  
- TF-IDF Word2Vec (GloVe model)  
- Separate vectors for Q1 and Q2  
- Vectors → 96 dimensions each  

---

## 🧾 Final Dataframe
- Preprocessed dataframe  
- NLP features dataframe  
- Q1 vector dataframe  
- Q2 vector dataframe  
- **Final dataframe → 218 features**  

---

## 📚 Train-Test Split
- Train: 70%  
- Test: 30%  

---

## 🤖 Models & Results
- **Random Model** → Log loss: `0.89` (worst case)  
- **Logistic Regression** → Log loss: `0.43`  
- **SVM** → Log loss: `0.44`  
- **XGBoost** → Log loss: `0.36` (improved: `0.33`)  
- **Decision Tree** → Log loss: `0.41`  
- **Random Forest** → Log loss: `0.43`  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/quora-question-pair-similarity.git
   cd quora-question-pair-similarity
