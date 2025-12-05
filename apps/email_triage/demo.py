#!/usr/bin/env python3
"""
IIIT Email Triage System - Demonstration Script
================================================
This script demonstrates the email triage system with various examples
covering different categories, urgency levels, and tones.
"""

import json
from triage import classify_email

# Test cases for different categories, urgency levels, and tones
test_emails = [
    # Attendance - Critical/Frustrated
    "My attendance is showing 50% in DS but I NEVER missed a single class!!! This is URGENT, exams are in 2 days!",
    
    # Marks/Reeval - High/Angry  
    "I got only 5 marks in ML midsem but I wrote everything correctly! This is completely unfair. I demand a recheck immediately!",
    
    # Hostel - Critical/Angry
    "AC is broken for 5 DAYS now!! Room is like an oven at 45 degrees! Nobody cares about students here! Fix this NOW!!!",
    
    # Finance - Medium/Polite
    "Dear Sir, I kindly request information about my pending scholarship. When will it be credited to my account? Thank you.",
    
    # Certificate - High/Neutral
    "I need a bonafide certificate urgently for my visa application. The deadline is tomorrow. Please help.",
    
    # Technical - High/Frustrated
    "WiFi has been down in OBH for 3 days! Cannot attend online classes, cannot submit assignments. Very frustrated!",
    
    # General - Low/Polite
    "Could you please tell me the library timings during exam week? Thank you for your help.",
    
    # Assignment - Medium/Confused
    "I am confused about problem 3 in the DBMS assignment. The question is not clear. Can someone clarify?",
    
    # Medical - Critical/Polite
    "I was hospitalized due to dengue last week. Please excuse my absence and accept my late assignment submission. Attaching medical certificate.",
    
    # Mess - Medium/Angry
    "The food quality is terrible! Found insects in dal today. This is unacceptable and unhygienic!"
]


def main():
    print("=" * 80)
    print("  IIIT EMAIL TRIAGE SYSTEM - DEMONSTRATION")
    print("  Classifying emails by Category, Urgency, and Tone")
    print("=" * 80)
    
    results = []
    
    for i, email in enumerate(test_emails, 1):
        result = classify_email(email)
        
        cat = result["predictions"]["category"]["ensemble"]
        cat_conf = result["model_comparison"]["category"]["lr_confidence"]
        
        urg = result["predictions"]["urgency"]["ensemble"]
        urg_conf = result["model_comparison"]["urgency"]["lr_confidence"]
        
        tone = result["predictions"]["tone"]["ensemble"]
        tone_conf = result["model_comparison"]["tone"]["lr_confidence"]
        
        print(f"\n{'─' * 80}")
        print(f"Example {i}:")
        print(f"{'─' * 80}")
        
        # Truncate email for display
        display_email = email[:75] + "..." if len(email) > 75 else email
        print(f"📧 Email: {display_email}")
        print()
        print(f"   📁 Category: {cat:20} (confidence: {cat_conf:.1%})")
        print(f"   ⚠️  Urgency:  {urg:20} (confidence: {urg_conf:.1%})")
        print(f"   💬 Tone:     {tone:20} (confidence: {tone_conf:.1%})")
        
        # Store for summary
        results.append({
            'email': email[:50] + "...",
            'category': cat,
            'urgency': urg,
            'tone': tone
        })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'#':<3} {'Category':<20} {'Urgency':<12} {'Tone':<15} {'Email (truncated)':<30}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<3} {r['category']:<20} {r['urgency']:<12} {r['tone']:<15} {r['email'][:30]}")
    
    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    # Usage instructions
    print("""
HOW TO USE THE EMAIL TRIAGE SYSTEM:
───────────────────────────────────

1. Single Email Classification:
   $ python triage.py --text "Your email text here"

2. Batch Classification from File:
   $ python triage.py --file emails.txt

3. Retrain Models:
   $ python triage.py --train

4. Run Demo:
   $ python demo.py

Output includes:
- Category (11 classes): Attendance, Academic/Course, Marks/Reeval, Assignment,
  Hostel, Mess, Finance/Fee, Certificate, Medical, Technical, General
  
- Urgency (4 levels): Critical, High, Medium, Low

- Tone (6 types): Angry, Frustrated, Confused, Polite, Neutral, Appreciative
""")


if __name__ == "__main__":
    main()
