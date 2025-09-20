#!/usr/bin/env python3

"""
Example usage of the AI-Powered Customer Support Ticket Classifier
"""

import json
import subprocess
import sys


def classify_ticket(title, description, customer_email=None):
    """Classify a ticket using the API"""
    data = {
        "title": title,
        "description": description
    }
    if customer_email:
        data["customer_email"] = customer_email
    
    json_data = json.dumps(data)
    
    try:
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', 'http://localhost:8000/classify',
            '-H', 'Content-Type: application/json',
            '-d', json_data
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": result.stderr}
            
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run example classifications"""
    print("🤖 AI-Powered Customer Support Ticket Classifier")
    print("=" * 55)
    print("This system automatically classifies support tickets by:")
    print("• Priority: Urgent, High, Medium, Low")
    print("• Department: Tech Support, Billing, Sales")
    print()
    
    # Example tickets
    examples = [
        {
            "title": "Website crashed!",
            "description": "The entire website is down and customers can't place orders",
            "customer_email": "admin@ecommerce.com"
        },
        {
            "title": "Double charged",
            "description": "I was charged twice for my subscription this month",
            "customer_email": "customer@gmail.com"
        },
        {
            "title": "Pricing question",
            "description": "What are the costs for the enterprise plan with 100 users?",
            "customer_email": "decision.maker@bigcorp.com"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"📩 Example {i}:")
        print(f"From: {example['customer_email']}")
        print(f"Subject: {example['title']}")
        print(f"Message: {example['description']}")
        print()
        
        # Classify the ticket
        result = classify_ticket(
            example['title'],
            example['description'],
            example['customer_email']
        )
        
        if 'error' not in result:
            print("🔍 Classification Results:")
            print(f"   Priority: {result['predicted_priority']} ({result['priority_confidence']:.1%} confidence)")
            print(f"   Route to: {result['predicted_department']} ({result['department_confidence']:.1%} confidence)")
            
            # Suggest action based on priority
            if result['predicted_priority'] == 'Urgent':
                print("   🚨 Action: Immediate escalation required!")
            elif result['predicted_priority'] == 'High':
                print("   ⚡ Action: Handle within 2 hours")
            elif result['predicted_priority'] == 'Medium':
                print("   📋 Action: Handle within 24 hours")
            else:
                print("   📝 Action: Handle within 3 business days")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 55)
        print()


if __name__ == "__main__":
    # Check if server is running
    try:
        result = subprocess.run([
            'curl', '-s', 'http://localhost:8000/health'
        ], capture_output=True, text=True, timeout=3)
        
        if result.returncode != 0:
            print("❌ Error: API server is not running!")
            print("Please start the server first with: python app/simple_main.py")
            sys.exit(1)
            
    except Exception:
        print("❌ Error: Cannot connect to API server!")
        print("Please start the server first with: PYTHONPATH=. python app/simple_main.py")
        sys.exit(1)
    
    main()