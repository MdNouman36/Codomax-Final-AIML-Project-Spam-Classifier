# Codomax-Final-AIML-Project-Spam-Classifier
# Final AI & ML Capstone Project: Spam Email & SMS Classifier
**Internship:** AI & ML Virtual Internship at Codomax Digital Solutions  
**Intern:** Mohammed Nouman M N  
**Module:** Module 06 — Final AI & ML Project (07/12/2026 – 10/12/2026)  

---

## 1. Project Overview
This project builds an end-to-end Natural Language Processing (NLP) and Machine Learning classification pipeline that detects whether incoming emails or SMS messages are **Spam** or **Ham (Legitimate)**.

---

## 2. Tech Stack & Pipeline Architecture
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Feature Extraction:** TF-IDF Vectorizer (`TfidfVectorizer` with English stop-word removal)
* **Machine Learning Model:** Multinomial Naive Bayes (`MultinomialNB`)
* **Evaluation & Visualization:** Scikit-learn (`accuracy_score`, `classification_report`, `ConfusionMatrixDisplay`) & Matplotlib

### Workflow Steps:
1. **Data Preparation:** Load labeled text messages (`ham` vs. `spam`) into a Pandas DataFrame and encode target labels.
2. **Text Vectorization:** Convert raw text messages into numerical TF-IDF feature matrices.
3. **Train-Test Split:** Split data into training (75%) and testing (25%) subsets.
4. **Model Training:** Train a `MultinomialNB` classifier on the vectorized training features.
5. **Evaluation & Inference:** Evaluate test accuracy, generate a confusion matrix plot, and classify custom real-time messages.

---

## 3. How to Run
```bash
pip install pandas scikit-learn matplotlib
python spam_classifier.py
