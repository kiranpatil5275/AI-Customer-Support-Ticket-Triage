"""
Generate a realistic synthetic dataset of support tickets.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Categories and priorities
CATEGORIES = ["billing", "technical", "account", "product"]
PRIORITIES = ["low", "medium", "high", "urgent"]

# Realistic ticket templates for each category
TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month. Please refund the extra amount.",
        "My invoice shows incorrect tax calculation. Can you fix it?",
        "I need a copy of last month's invoice for accounting purposes.",
        "Payment failed but money was deducted from my account. Please help.",
        "How do I update my credit card details for auto-renewal?",
        "I cancelled my plan but still got billed. Requesting refund.",
        "There is a discrepancy in the billing amount for enterprise plan.",
        "Can you explain the extra charges on my recent invoice?",
        "I want to change from monthly to annual billing.",
        "My payment method expired. How do I update it without service interruption?",
        "Received a bill for a service I never signed up for.",
        "Need GST invoice for my company purchase.",
        "The promotional discount was not applied to my bill.",
        "I was charged for premium features I didn't use.",
        "Please pause my subscription and stop billing temporarily."
    ],
    "technical": [
        "The application keeps crashing whenever I try to upload a large file.",
        "I am unable to login. Getting error 500 internal server error.",
        "Dashboard is loading very slowly for the past two days.",
        "API integration is failing with authentication error.",
        "Mobile app freezes on the home screen after latest update.",
        "I cannot export reports in PDF format. Button does nothing.",
        "Getting CORS error when calling the public API.",
        "Search functionality is not returning correct results.",
        "Webhooks are not being triggered for new events.",
        "SSO login with Google is broken since yesterday.",
        "File upload progress bar stuck at 99%.",
        "Real-time notifications are delayed by several minutes.",
        "The system is showing old data even after refresh.",
        "Getting timeout errors during peak hours.",
        "Chrome extension is not syncing with the web app."
    ],
    "account": [
        "I forgot my password and the reset email is not arriving.",
        "How can I change the email address linked to my account?",
        "I need to add a new team member with admin privileges.",
        "Please delete my account permanently as per GDPR.",
        "I am unable to verify my phone number for 2FA.",
        "My account was locked after multiple failed login attempts.",
        "How do I transfer ownership of the account to another person?",
        "Need to update company name and address in profile.",
        "I want to enable two-factor authentication.",
        "Someone else has access to my account. Please secure it.",
        "How can I view login history of my account?",
        "I need to merge two accounts into one.",
        "Account recovery options are not working.",
        "Please remove the old admin who left the company.",
        "I cannot change my username. Is it permanent?"
    ],
    "product": [
        "Does the free plan include access to the API?",
        "What is the difference between Pro and Enterprise plans?",
        "Is there a dark mode available in the application?",
        "Can I integrate this tool with Slack and Microsoft Teams?",
        "How many users can I add on the current plan?",
        "Is offline mode supported on the mobile app?",
        "Do you offer white-label solution for agencies?",
        "What are the data retention policies?",
        "Can I customize the dashboard widgets?",
        "Is there a public roadmap for upcoming features?",
        "Does the tool support multi-language interface?",
        "How does the AI feature work and is it included?",
        "Can I export all my data in bulk?",
        "Is there a desktop application available?",
        "What is the uptime SLA for the paid plans?"
    ]
}

# Additional variation words to make tickets more realistic
EXTRA_PHRASES = [
    " This is urgent.",
    " Please help as soon as possible.",
    " I have been waiting for 3 days.",
    " Kindly look into this matter.",
    " Looking forward to your response.",
    " This is affecting my business.",
    " I am a premium customer.",
    " Please escalate this issue.",
    " Thank you in advance.",
    ""
]

def generate_ticket_text(category: str) -> str:
    base = random.choice(TEMPLATES[category])
    extra = random.choice(EXTRA_PHRASES)
    # Sometimes add ticket ID style references
    if random.random() < 0.3:
        base += f" Reference: TKT-{random.randint(10000, 99999)}"
    return (base + extra).strip()

def generate_dataset(n_samples: int = 1000) -> pd.DataFrame:
    data = []
    
    for i in range(1, n_samples + 1):
        category = random.choice(CATEGORIES)
        # Priority distribution: more medium/high than urgent
        priority = random.choices(
            PRIORITIES,
            weights=[0.25, 0.35, 0.25, 0.15],
            k=1
        )[0]
        
        ticket_text = generate_ticket_text(category)
        
        # Resolution time roughly correlated with priority
        if priority == "urgent":
            resolution = round(np.random.uniform(0.5, 8), 1)
        elif priority == "high":
            resolution = round(np.random.uniform(4, 24), 1)
        elif priority == "medium":
            resolution = round(np.random.uniform(12, 48), 1)
        else:
            resolution = round(np.random.uniform(24, 96), 1)
        
        data.append({
            "ticket_id": f"TKT-{10000 + i}",
            "ticket_text": ticket_text,
            "category": category,
            "priority": priority,
            "resolution_time_hours": resolution
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "support_tickets.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = generate_dataset(1000)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} tickets")
    print(f"📁 Saved to: {output_path}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nPriority distribution:")
    print(df["priority"].value_counts())
    print("\nSample tickets:")
    print(df.head(3).to_string())
