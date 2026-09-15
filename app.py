"""
Streamlit Dashboard for AI Customer Support Ticket Triage
"""

import streamlit as st
import joblib
import re
from pathlib import Path
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Ticket Triage System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.joblib"

# Simple stopwords (same as training)
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
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


@st.cache_resource
def load_models():
    """Load trained models (cached)."""
    if not CATEGORY_MODEL_PATH.exists() or not PRIORITY_MODEL_PATH.exists():
        return None, None
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    return category_model, priority_model


def get_routing_message(category: str, priority: str) -> str:
    """Generate a simple routing recommendation."""
    queue_map = {
        "billing": "Billing & Payments Queue",
        "technical": "Technical Support Queue",
        "account": "Account Management Queue",
        "product": "Product & Feature Requests Queue"
    }
    queue = queue_map.get(category, "General Support Queue")
    
    if priority in ["urgent", "high"]:
        return f"🚨 **Escalate to {queue}** (Priority: {priority.upper()})"
    else:
        return f"✅ **Route to {queue}** (Priority: {priority})"


def main():
    st.title("🎫 AI Customer Support Ticket Triage")
    st.markdown("Predict **Category** + **Urgency** and automatically route tickets.")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system uses machine learning to:
        - Classify tickets into **billing / technical / account / product**
        - Predict **priority** (low / medium / high / urgent)
        - Flag low-confidence predictions for human review
        """)
        st.markdown("---")
        st.markdown("**Confidence Threshold:** 0.65")
        st.markdown("Predictions below this are flagged for review.")
    
    # Load models
    category_model, priority_model = load_models()
    
    if category_model is None or priority_model is None:
        st.error("⚠️ Models not found!")
        st.info("""
        Please train the models first:
        ```bash
        cd ticket_triage_project
        python src/generate_data.py
        python src/train_models.py
        ```
        Then refresh this page.
        """)
        return
    
    st.success("✅ Models loaded successfully")
    
    # Main input area
    st.subheader("📝 Enter Support Ticket")
    
    # Example tickets for quick testing
    examples = {
        "Select an example...": "",
        "Billing issue": "I was charged twice for my subscription this month. Please refund the extra amount immediately.",
        "Technical crash": "The application keeps crashing whenever I try to upload a large file. This is urgent.",
        "Account access": "I forgot my password and the reset email is not arriving. Please help me recover my account.",
        "Product question": "Does the free plan include access to the API and how many users can I add?"
    }
    
    selected_example = st.selectbox("Or choose a sample ticket:", list(examples.keys()))
    
    default_text = examples[selected_example]
    
    ticket_text = st.text_area(
        "Ticket Text",
        value=default_text,
        height=150,
        placeholder="Paste the customer support ticket here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🔍 Predict & Route", type="primary", use_container_width=True)
    
    if predict_btn:
        if not ticket_text.strip():
            st.warning("Please enter a ticket text.")
            return
        
        with st.spinner("Analyzing ticket..."):
            cleaned = clean_text(ticket_text)
            
            # Category prediction
            cat_pred = category_model.predict([cleaned])[0]
            cat_proba = category_model.predict_proba([cleaned])[0]
            cat_conf = float(cat_proba.max())
            cat_classes = category_model.classes_
            
            # Priority prediction
            pri_pred = priority_model.predict([cleaned])[0]
            pri_proba = priority_model.predict_proba([cleaned])[0]
            pri_conf = float(pri_proba.max())
            pri_classes = priority_model.classes_
            
            overall_conf = min(cat_conf, pri_conf)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("Category", cat_pred.upper())
            with m2:
                st.metric("Category Confidence", f"{cat_conf:.1%}")
            with m3:
                st.metric("Priority", pri_pred.upper())
            with m4:
                st.metric("Priority Confidence", f"{pri_conf:.1%}")
            
            # Routing message
            st.markdown("### 🚀 Routing Recommendation")
            routing_msg = get_routing_message(cat_pred, pri_pred)
            st.info(routing_msg)
            
            # Low confidence flag
            CONFIDENCE_THRESHOLD = 0.65
            if overall_conf < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ **Low Confidence Alert** ({overall_conf:.1%} < {CONFIDENCE_THRESHOLD:.0%})\n\n"
                    "This ticket has been flagged for **Human Review**."
                )
            else:
                st.success("✅ High confidence prediction — safe to auto-route.")
            
            # Detailed probability breakdown
            with st.expander("🔎 View detailed probability scores"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Category Probabilities**")
                    cat_df = pd.DataFrame({
                        "Category": cat_classes,
                        "Probability": cat_proba
                    }).sort_values("Probability", ascending=False)
                    st.dataframe(
                        cat_df.style.format({"Probability": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col_b:
                    st.markdown("**Priority Probabilities**")
                    pri_df = pd.DataFrame({
                        "Priority": pri_classes,
                        "Probability": pri_proba
                    }).sort_values("Probability", ascending=False)
                    st.dataframe(
                        pri_df.style.format({"Probability": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Cleaned text preview
            with st.expander("🧹 Cleaned text used by model"):
                st.code(cleaned if cleaned else "(empty after cleaning)")


if __name__ == "__main__":
    main()
