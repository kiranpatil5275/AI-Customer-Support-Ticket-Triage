# 🎫 AI Customer Support Ticket Triage

A complete medium-level AI project that automatically classifies support tickets into categories and predicts urgency/priority, then routes them to the correct queue.

---

## 📌 Project Overview

**Goal:** Build an AI system that reads incoming support tickets, predicts their **category** and **urgency**, and routes them to the appropriate support queue.

### Features
- Synthetic dataset of 1000 realistic support tickets
- Text cleaning + TF-IDF features
- Two separate models:
  - Multiclass classifier for **Category** (billing / technical / account / product)
  - Classifier for **Priority** (low / medium / high / urgent)
- Full evaluation (F1-score, confusion matrix, precision/recall)
- Beautiful Streamlit dashboard
- Low-confidence predictions (< 0.65) are flagged for human review
- Simple routing recommendations

---

## 🗂️ Project Structure

```
ticket_triage_project/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   └── support_tickets.csv # Generated dataset
├── models/
│   ├── category_model.joblib
│   └── priority_model.joblib
└── src/
    ├── generate_data.py    # Create synthetic data
    └── train_models.py     # Train & evaluate models
```

---

## ⚙️ How to Run (Step-by-Step)

### 1. Install dependencies
```bash
cd ticket_triage_project
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python src/generate_data.py
```
This creates `data/support_tickets.csv` with 1000 tickets.

### 3. Train the models
```bash
python src/train_models.py
```
This will:
- Clean the text
- Train Category + Priority models
- Print classification reports & confusion matrices
- Save models to the `models/` folder

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```
The app will open in your browser (usually http://localhost:8501).

---

## 🧪 How to Use the Dashboard

1. Paste a support ticket (or select a sample)
2. Click **Predict & Route**
3. See:
   - Predicted Category + Confidence
   - Predicted Priority + Confidence
   - Routing recommendation
   - Low-confidence warning (if any)
   - Detailed probability scores

---

## 📊 Model Details

| Model              | Algorithm              | Features     |
|--------------------|------------------------|--------------|
| Category Classifier| Logistic Regression    | TF-IDF (1-2 grams) |
| Priority Classifier| Random Forest          | TF-IDF (1-2 grams) |

**Confidence Threshold:** 0.65  
Any prediction below this threshold is flagged for human review.

---

## 📦 Requirements

- Python 3.8+
- pandas
- scikit-learn
- streamlit
- joblib
- nltk (optional, has fallback)

---

## 📝 Notes for Submission / Internship

- All code is modular and well-commented
- Fully offline (no paid APIs)
- Easy to extend with real data later
- You can take screenshots of:
  - Classification reports from training
  - Streamlit dashboard predictions
  - Low-confidence flags

---

## 🚀 Next Improvements (Optional)

- Add hyperparameter tuning with GridSearchCV
- Use sentence-transformers embeddings
- Add a simple feedback loop for wrong predictions
- Deploy on Streamlit Cloud or Hugging Face Spaces

---

**Made for IBM / Mindenious AI Internship Capstone Project**
