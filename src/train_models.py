"""
Train Category and Priority models for Support Ticket Triage.
"""

import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Try to use NLTK stopwords, fallback to simple list
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now"
    }


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def load_and_prepare_data(data_path: Path):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} tickets")
    
    df["clean_text"] = df["ticket_text"].apply(clean_text)
    
    # Remove any empty texts after cleaning
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
    print(f"After cleaning: {len(df)} tickets")
    
    return df


def train_category_model(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("TRAINING CATEGORY CLASSIFIER")
    print("="*60)
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=1.0
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\nClassification Report (Category):")
    print(classification_report(y_test, y_pred, digits=3))
    
    print("Confusion Matrix (Category):")
    print(confusion_matrix(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nWeighted F1-Score: {f1:.4f}")
    
    return model


def train_priority_model(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("TRAINING PRIORITY / URGENCY CLASSIFIER")
    print("="*60)
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report (Priority):")
    print(classification_report(y_test, y_pred, digits=3))
    
    print("Confusion Matrix (Priority):")
    print(confusion_matrix(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nWeighted F1-Score: {f1:.4f}")
    
    return model


def main():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "support_tickets.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data if not exists
    if not data_path.exists():
        print("Data not found. Generating synthetic dataset...")
        from generate_data import generate_dataset
        df = generate_dataset(1000)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Saved to {data_path}")
    
    df = load_and_prepare_data(data_path)
    
    X = df["clean_text"]
    y_category = df["category"]
    y_priority = df["priority"]
    
    # Split once and reuse for both models
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        X, y_category, y_priority,
        test_size=0.2,
        random_state=42,
        stratify=y_category
    )
    
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Train both models
    category_model = train_category_model(X_train, X_test, y_cat_train, y_cat_test)
    priority_model = train_priority_model(X_train, X_test, y_pri_train, y_pri_test)
    
    # Save models
    category_path = models_dir / "category_model.joblib"
    priority_path = models_dir / "priority_model.joblib"
    
    joblib.dump(category_model, category_path)
    joblib.dump(priority_model, priority_path)
    
    print("\n" + "="*60)
    print("✅ MODELS SAVED SUCCESSFULLY")
    print("="*60)
    print(f"Category model → {category_path}")
    print(f"Priority model → {priority_path}")
    
    # Quick sanity check
    sample = "I was charged twice and need a refund urgently"
    clean = clean_text(sample)
    cat_pred = category_model.predict([clean])[0]
    cat_conf = category_model.predict_proba([clean]).max()
    pri_pred = priority_model.predict([clean])[0]
    pri_conf = priority_model.predict_proba([clean]).max()
    
    print(f"\nSample prediction:")
    print(f"Text: {sample}")
    print(f"Category: {cat_pred} (confidence: {cat_conf:.2f})")
    print(f"Priority: {pri_pred} (confidence: {pri_conf:.2f})")


if __name__ == "__main__":
    main()
