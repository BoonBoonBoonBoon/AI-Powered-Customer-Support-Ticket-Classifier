import pandas as pd
import numpy as np
from typing import List, Dict

def generate_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Generate a sample dataset for training the ticket classifier"""
    
    np.random.seed(42)
    
    # Sample ticket titles and descriptions for different departments and priorities
    tech_support_data = [
        {"title": "Website not loading", "description": "The website shows a 404 error when I try to access my account", "department": "Tech Support", "priority": "High"},
        {"title": "Password reset issue", "description": "I can't reset my password, the email is not coming through", "department": "Tech Support", "priority": "Medium"},
        {"title": "App crashes frequently", "description": "The mobile app keeps crashing when I try to upload files", "department": "Tech Support", "priority": "High"},
        {"title": "Slow loading times", "description": "The dashboard takes forever to load, it's affecting my productivity", "department": "Tech Support", "priority": "Medium"},
        {"title": "Server down", "description": "Unable to connect to the server, getting timeout errors", "department": "Tech Support", "priority": "Urgent"},
        {"title": "API not responding", "description": "Getting 500 internal server error when making API calls", "department": "Tech Support", "priority": "Urgent"},
        {"title": "Database connection error", "description": "Cannot connect to database, all services are affected", "department": "Tech Support", "priority": "Urgent"},
        {"title": "Login problems", "description": "Cannot log into my account, shows invalid credentials", "department": "Tech Support", "priority": "High"},
        {"title": "Feature not working", "description": "The export function is not working properly", "department": "Tech Support", "priority": "Low"},
        {"title": "Browser compatibility", "description": "Site doesn't work properly on Safari browser", "department": "Tech Support", "priority": "Low"},
    ]
    
    billing_data = [
        {"title": "Incorrect billing amount", "description": "I was charged twice for my subscription this month", "department": "Billing", "priority": "High"},
        {"title": "Payment failed", "description": "My credit card payment was declined but money was deducted", "department": "Billing", "priority": "High"},
        {"title": "Refund request", "description": "I want to request a refund for unused services", "department": "Billing", "priority": "Medium"},
        {"title": "Subscription cancellation", "description": "I need to cancel my subscription immediately", "department": "Billing", "priority": "Medium"},
        {"title": "Invoice missing", "description": "I haven't received my invoice for last month", "department": "Billing", "priority": "Low"},
        {"title": "Payment method update", "description": "Need to update my credit card information", "department": "Billing", "priority": "Low"},
        {"title": "Billing address change", "description": "Please update my billing address", "department": "Billing", "priority": "Low"},
        {"title": "Tax information", "description": "Need tax documentation for business expenses", "department": "Billing", "priority": "Medium"},
        {"title": "Overcharged account", "description": "My account shows charges I didn't authorize", "department": "Billing", "priority": "Urgent"},
        {"title": "Plan upgrade billing", "description": "Questions about pricing for plan upgrade", "department": "Billing", "priority": "Medium"},
    ]
    
    sales_data = [
        {"title": "Product pricing inquiry", "description": "What are the pricing options for enterprise plans?", "department": "Sales", "priority": "Medium"},
        {"title": "Demo request", "description": "I would like to schedule a product demonstration", "department": "Sales", "priority": "High"},
        {"title": "Feature comparison", "description": "Can you compare your product with competitors?", "department": "Sales", "priority": "Low"},
        {"title": "Custom solution needed", "description": "We need a custom integration for our business", "department": "Sales", "priority": "High"},
        {"title": "Enterprise inquiry", "description": "Interested in enterprise licensing and support", "department": "Sales", "priority": "High"},
        {"title": "Trial extension", "description": "Can we extend our trial period?", "department": "Sales", "priority": "Medium"},
        {"title": "Volume discount", "description": "Do you offer discounts for bulk purchases?", "department": "Sales", "priority": "Medium"},
        {"title": "Partnership opportunity", "description": "Interested in becoming a reseller partner", "department": "Sales", "priority": "Medium"},
        {"title": "Product roadmap", "description": "What new features are planned for next year?", "department": "Sales", "priority": "Low"},
        {"title": "Contract negotiation", "description": "Need to discuss contract terms and conditions", "department": "Sales", "priority": "High"},
    ]
    
    # Create base templates
    all_templates = tech_support_data + billing_data + sales_data
    
    # Generate variations by modifying templates
    dataset = []
    
    for i in range(n_samples):
        template = np.random.choice(all_templates)
        
        # Add some variation to the text
        variations = [
            template["title"],
            f"URGENT: {template['title']}",
            f"Help needed: {template['title']}",
            f"Issue with {template['title'].lower()}",
            template["title"].upper(),
        ]
        
        desc_variations = [
            template["description"],
            f"Please help! {template['description']}",
            f"{template['description']} This is affecting my business.",
            f"{template['description']} Can someone assist?",
            f"{template['description']} Thanks in advance.",
        ]
        
        # Randomly adjust priority sometimes
        priority = template["priority"]
        if np.random.random() < 0.1:  # 10% chance to change priority
            priorities = ["Urgent", "High", "Medium", "Low"]
            priority = np.random.choice(priorities)
        
        ticket = {
            "title": np.random.choice(variations),
            "description": np.random.choice(desc_variations),
            "department": template["department"],
            "priority": priority,
            "customer_email": f"customer{i}@example.com"
        }
        
        dataset.append(ticket)
    
    return pd.DataFrame(dataset)


def save_sample_dataset(output_path: str, n_samples: int = 1000) -> None:
    """Generate and save a sample dataset"""
    df = generate_sample_dataset(n_samples)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset with {len(df)} tickets saved to {output_path}")
    print(f"Department distribution:\n{df['department'].value_counts()}")
    print(f"Priority distribution:\n{df['priority'].value_counts()}")


if __name__ == "__main__":
    save_sample_dataset("data/sample_tickets.csv", 1000)