#!/usr/bin/env python3

"""
End-to-end test of the ticket classifier API
"""

import json
import sys
import time
import subprocess


# Helper function (not a pytest test) to call API
def call_api_endpoint(title, description, customer_email=None):
    """Test a single API endpoint"""
    data = {
        "title": title,
        "description": description
    }
    if customer_email:
        data["customer_email"] = customer_email
    
    json_data = json.dumps(data)
    
    try:
        # Use curl to test the API
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', 'http://localhost:8000/classify',
            '-H', 'Content-Type: application/json',
            '-d', json_data
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            return response
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def main():
    """Run comprehensive end-to-end tests"""
    print("🚀 Starting End-to-End API Tests")
    print("=" * 60)
    
    # Test cases covering different scenarios
    test_cases = [
        {
            "title": "Server Critical Outage",
            "description": "All servers are down, entire system offline, urgent attention needed",
            "expected_priority": "Urgent",
            "expected_department": "Tech Support",
            "customer_email": "enterprise@bigcorp.com"
        },
        {
            "title": "Payment Processing Failed",
            "description": "Credit card payment failed but money was charged to customer account",
            "expected_priority": "High",
            "expected_department": "Billing"
        },
        {
            "title": "Product Demo Request",
            "description": "Interested in enterprise features, need demonstration for decision makers",
            "expected_priority": "High",
            "expected_department": "Sales",
            "customer_email": "sales.lead@company.com"
        },
        {
            "title": "How to change password",
            "description": "I forgot my password and need help resetting it",
            "expected_priority": "Medium",
            "expected_department": "Tech Support"
        },
        {
            "title": "Invoice question",
            "description": "I have a question about last month's invoice charges",
            "expected_priority": "Low",
            "expected_department": "Billing"
        },
        {
            "title": "Feature comparison",
            "description": "Can you provide information about your pricing plans?",
            "expected_priority": "Medium",
            "expected_department": "Sales"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['title']}")
        print("-" * 40)
        response = call_api_endpoint(
            test_case['title'],
            test_case['description'],
            test_case.get('customer_email')
        )
        
        if response:
            # Display results
            print(f"Input:")
            print(f"  Title: {test_case['title']}")
            print(f"  Description: {test_case['description']}")
            if test_case.get('customer_email'):
                print(f"  Email: {test_case['customer_email']}")
            
            print(f"\nClassification Results:")
            print(f"  Priority: {response['predicted_priority']} (confidence: {response['priority_confidence']:.2f})")
            print(f"  Department: {response['predicted_department']} (confidence: {response['department_confidence']:.2f})")
            
            # Check expectations
            priority_match = response['predicted_priority'] == test_case['expected_priority']
            dept_match = response['predicted_department'] == test_case['expected_department']
            
            print(f"\nExpectation Check:")
            print(f"  Priority: {'✅' if priority_match else '❌'} (expected: {test_case['expected_priority']})")
            print(f"  Department: {'✅' if dept_match else '❌'} (expected: {test_case['expected_department']})")
            
            results.append({
                'test_case': i,
                'priority_match': priority_match,
                'department_match': dept_match,
                'response': response
            })
        else:
            print("❌ API request failed")
            results.append({
                'test_case': i,
                'priority_match': False,
                'department_match': False,
                'response': None
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    total_tests = len(results)
    priority_correct = sum(1 for r in results if r['priority_match'])
    dept_correct = sum(1 for r in results if r['department_match'])
    api_success = sum(1 for r in results if r['response'] is not None)
    
    print(f"Total Test Cases: {total_tests}")
    print(f"API Success Rate: {api_success}/{total_tests} ({api_success/total_tests*100:.1f}%)")
    print(f"Priority Accuracy: {priority_correct}/{total_tests} ({priority_correct/total_tests*100:.1f}%)")
    print(f"Department Accuracy: {dept_correct}/{total_tests} ({dept_correct/total_tests*100:.1f}%)")
    
    overall_success = api_success == total_tests and priority_correct >= total_tests * 0.8 and dept_correct >= total_tests * 0.8
    
    if overall_success:
        print("\n🎉 End-to-End Tests PASSED! System is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please review the results above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)