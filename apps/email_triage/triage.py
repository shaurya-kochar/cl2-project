#!/usr/bin/env python3

import sys
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.append(str(PROJECT_ROOT / 'src'))
from preprocess import clean_text, tokenize_text
from features import FeatureExtractor


CATEGORIES = [
    'Attendance', 'Academic/Course', 'Marks/Reeval', 'Assignment',
    'Hostel', 'Mess', 'Finance/Fee', 'Certificate', 'Medical', 
    'Technical', 'General'
]

URGENCY = ['Critical', 'High', 'Medium', 'Low']
TONE = ['Angry', 'Frustrated', 'Confused', 'Polite', 'Neutral', 'Appreciative']

URGENCY_KEYWORDS = {
    'Critical': ['urgent', 'asap', 'immediately', 'emergency', 'critical'],
    'High': ['soon', 'important', 'deadline', 'today', 'tomorrow'],
    'Medium': ['please', 'need', 'help', 'issue'],
    'Low': ['when', 'could', 'would', 'possible']
}

EMAIL_TEMPLATES = {
    'Attendance': [
        "Hi, my attendance for {course} is showing {percent}% but I attended all classes. Please check.",
        "Dear Sir, I need attendance condonation for {course}. I was sick for 2 weeks.",
        "My attendance is not updated for {course}. Kindly update it ASAP.",
        "Attendance portal shows wrong attendance count for {course}. Need correction.",
        "I have shortage of attendance in {course} due to medical reasons. Please consider.",
        "The biometric attendance machine didn't record my entry today for {course} class.",
        "I was marked absent in {course} but I was present in class. Please verify.",
        "Need attendance relaxation for {course}. I had family emergency.",
        "My proxy attendance was caught. I accept my mistake. Please give me one chance.",
        "Attendance percentage is below 75% in {course}. What are my options?"
    ],
    'Academic/Course': [
        "I want to drop {course}. What's the procedure?",
        "Can I audit {course} instead of credit? Please advise.",
        "The {course} prerequisites are not clear. Can you clarify?",
        "I want to change my branch from CSE to ECE. What is the process?",
        "Can I take extra courses this semester? I want to do {course}.",
        "The syllabus for {course} is not uploaded on Moodle.",
        "I need the reading list and textbooks for {course}.",
        "When will the course registration portal open for next semester?",
        "I want to do summer internship. What are the academic requirements?",
        "Can I take {course} as an elective? Is it available this semester?"
    ],
    'Marks/Reeval': [
        "I got {marks} in {course} midsem but my answers were correct. Please recheck.",
        "My marks for {course} assignment are very low. Can I get a re-evaluation?",
        "There's an error in my {course} grade calculation. Please verify.",
        "I want to apply for revaluation of my {course} answer sheet.",
        "My midsem marks are not updated on the portal for {course}.",
        "I think there is a totaling mistake in my {course} paper.",
        "Can I see my evaluated answer sheet for {course}?",
        "The quiz marks entered are wrong for {course}. I got {marks} not 5.",
        "My grade card shows F but I passed {course}. This is an error.",
        "I want to challenge the evaluation of question {num} in {course} endsem."
    ],
    'Assignment': [
        "The {course} assignment deadline is too tight. Can we get an extension?",
        "I'm confused about problem {num} in {course} assignment. Please clarify.",
        "My {course} assignment submission failed. Please accept late submission.",
        "The assignment {num} for {course} has ambiguous instructions.",
        "Can I submit the {course} assignment in PDF instead of word?",
        "I missed the assignment deadline for {course} due to internet issues.",
        "The plagiarism checker flagged my original work in {course} assignment.",
        "Can I resubmit the {course} assignment? I made some errors.",
        "When is the next assignment for {course} due?",
        "I need more time for {course} project submission. Team member is sick."
    ],
    'Hostel': [
        "My room AC is not working for 3 days. This is very urgent.",
        "There's a water leakage in room {num}. Please send maintenance.",
        "Can I change my room? My roommate is very noisy.",
        "The geyser in my bathroom is broken. No hot water for bathing.",
        "There are insects and cockroaches in my room. Need pest control.",
        "The room light and fan are not working. Need electrician urgently.",
        "My room lock is jammed. I can't lock my door properly.",
        "Internet is not working in my hostel wing for 2 days.",
        "The common room TV is broken. Can we get it repaired?",
        "I want to shift to a single room. Is vacancy available?"
    ],
    'Mess': [
        "The food quality in mess is terrible. We need better options.",
        "I have food allergies. Can I get special meal arrangements?",
        "Mess timings don't work with my lab schedule. Can this be changed?",
        "Found hair in the food today. This is very unhygienic.",
        "The mess food is always cold and stale. Please improve quality.",
        "I am vegetarian but no veg options available in dinner menu.",
        "I want to cancel my mess subscription. What is the refund policy?",
        "The mess card is not working. Please reset my account.",
        "Can we get more variety in breakfast menu? Same food daily.",
        "The kitchen and dining area is very dirty. Need cleaning."
    ],
    'Finance/Fee': [
        "I haven't received my scholarship yet. When will it be credited?",
        "There's an error in my fee receipt. Please correct it.",
        "Can I pay my hostel fee in installments? I'm facing financial issues.",
        "My scholarship application is pending since 2 months. Please expedite.",
        "I need fee concession due to family financial problems.",
        "The tuition fee has been deducted twice from my account.",
        "When is the last date to pay the semester fee?",
        "I want refund for the mess fee. I didn't use the mess.",
        "My education loan documents need college verification.",
        "How do I apply for the merit-cum-means scholarship?"
    ],
    'Certificate': [
        "I need a bonafide certificate urgently for visa application.",
        "How long does it take to get the degree certificate?",
        "I need transcript copies for MS application. What's the process?",
        "Can I get a character certificate for job application?",
        "I need provisional certificate. When will it be ready?",
        "My medium of instruction certificate is required for higher studies.",
        "How to get duplicate degree certificate? Original is lost.",
        "I need a no-dues certificate for final year clearance.",
        "Can I get migration certificate for transfer to another university?",
        "I need letter of recommendation from HOD for internship."
    ],
    'Medical': [
        "I'm very sick and need 3 days medical leave for {course}.",
        "Can you approve my medical certificate? I was hospitalized last week.",
        "I need attendance exemption due to dengue. Attaching medical reports.",
        "I have a doctor appointment during {course} class. Please excuse.",
        "I am not feeling well and cannot attend classes today.",
        "I need to visit hospital regularly for physiotherapy. Request permission.",
        "I have COVID symptoms. Should I isolate or attend classes?",
        "My medical insurance claim is pending. Please help with documents.",
        "I fainted in class today. Need medical emergency leave.",
        "I have chronic illness and need special seating arrangement in exams."
    ],
    'Technical': [
        "WiFi is not working in my hostel for 2 days. Very frustrated!",
        "I can't access Moodle. Getting server error repeatedly.",
        "My campus login credentials are not working. Please reset.",
        "The LAN connection in computer lab is very slow.",
        "I forgot my ERP password. How to reset it?",
        "The projector in classroom {num} is not working.",
        "Moodle is showing wrong course registrations for me.",
        "I cannot download my hall ticket from the portal.",
        "The online exam link is not opening. Please help urgently.",
        "My email ID is deactivated. I need it for placement applications."
    ],
    'General': [
        "Where is the admin office located?",
        "What are the library timings during exams?",
        "I lost my ID card. How do I get a duplicate?",
        "What is the procedure for campus gate pass?",
        "Where can I find the academic calendar?",
        "How to book the auditorium for cultural event?",
        "What are the visiting hours for parents?",
        "Where is the sports complex? I want to use gym.",
        "How to contact the placement cell?",
        "What are the rules for bringing guests to campus?"
    ]
}


def generate_synthetic_dataset(n=1100):
    """Generate a larger and more diverse synthetic email dataset"""
    data = []
    courses = ['COA', 'DS', 'DBMS', 'OS', 'AI', 'ML', 'CN', 'Maths', 'Physics', 'Chemistry', 
               'English', 'DSA', 'Algorithms', 'Networks', 'Security', 'Cloud', 'DevOps']
    
    # Tone modifiers to add variety
    angry_prefixes = ['This is unacceptable!', 'I am very angry!', 'What kind of service is this?',
                      'Nobody responds to emails here!', 'This is ridiculous!']
    frustrated_prefixes = ['Very frustrated!', 'I have been waiting for days!', 
                           'This is so annoying!', 'Why is this taking so long?']
    polite_prefixes = ['Dear Sir/Madam,', 'Respected Sir,', 'Hello,', 'Good morning,',
                       'I hope this email finds you well.']
    urgent_suffixes = ['This is very urgent!', 'Please respond ASAP!', 'Urgent help needed!',
                       'Time sensitive matter!', 'Emergency situation!']
    
    # Generate more balanced data for each category
    samples_per_category = n // len(CATEGORIES)
    
    for category in CATEGORIES:
        templates = EMAIL_TEMPLATES[category]
        
        for _ in range(samples_per_category):
            template = random.choice(templates)
            
            # Fill in template variables
            email_text = template.format(
                course=random.choice(courses),
                percent=random.randint(50, 80),
                marks=random.randint(5, 30),
                num=random.randint(1, 10)
            )
            
            # Determine urgency and tone based on content and random variation
            urgency = 'Medium'  # Default
            tone = 'Neutral'  # Default
            
            # Add modifiers to vary the tone
            modifier_roll = random.random()
            
            if modifier_roll < 0.15:  # 15% angry
                email_text = random.choice(angry_prefixes) + ' ' + email_text
                tone = 'Angry'
                urgency = random.choice(['Critical', 'High'])
            elif modifier_roll < 0.30:  # 15% frustrated
                email_text = random.choice(frustrated_prefixes) + ' ' + email_text
                tone = 'Frustrated'
                urgency = random.choice(['High', 'Medium'])
            elif modifier_roll < 0.60:  # 30% polite
                email_text = random.choice(polite_prefixes) + ' ' + email_text + ' Thank you.'
                tone = 'Polite'
                urgency = random.choice(['Medium', 'Low'])
            elif modifier_roll < 0.75:  # 15% urgent
                email_text = email_text + ' ' + random.choice(urgent_suffixes)
                urgency = 'Critical'
                tone = random.choice(['Frustrated', 'Neutral'])
            else:  # 25% neutral
                tone = 'Neutral'
                urgency = random.choice(URGENCY)
            
            # Override based on specific keywords
            if 'urgent' in email_text.lower() or 'asap' in email_text.lower() or 'immediately' in email_text.lower():
                urgency = 'Critical'
            elif 'emergency' in email_text.lower():
                urgency = 'Critical'
            elif 'please help' in email_text.lower() or 'need help' in email_text.lower():
                urgency = 'High'
            
            if '!' in email_text and email_text.count('!') >= 2:
                if tone not in ['Angry', 'Frustrated']:
                    tone = 'Frustrated'
            
            if 'thank' in email_text.lower() or 'kindly' in email_text.lower():
                tone = 'Polite'
            
            # Add some appreciative tones
            if random.random() < 0.05:
                email_text = email_text + ' Thanks for your help last time, it was very helpful!'
                tone = 'Appreciative'
                urgency = 'Low'
            
            # Add confused tones
            if 'confused' in email_text.lower() or 'not clear' in email_text.lower() or 'clarify' in email_text.lower():
                tone = 'Confused'
            
            data.append({
                'email_text': email_text,
                'category': category,
                'urgency': urgency,
                'tone': tone
            })
    
    df = pd.DataFrame(data)
    # Shuffle the dataframe
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def extract_email_features(texts):
    extractor = FeatureExtractor(tfidf_max_features=1000, tfidf_ngram=(1,2))
    extractor.fit_tfidf(texts)
    
    X_tfidf = extractor.transform_tfidf(texts).toarray()
    
    features = []
    for text in texts:
        tokens = tokenize_text(clean_text(text))
        
        urgency_score = sum(
            1 for word in tokens 
            for keywords in URGENCY_KEYWORDS.values()
            if word.lower() in keywords
        )
        
        punct_intensity = text.count('!') + text.count('?') * 0.5
        caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        negation_count = sum(1 for w in tokens if w.lower() in {'not', 'no', 'never'})
        politeness = sum(1 for w in tokens if w.lower() in {'please', 'kindly', 'thank'})
        
        features.append([
            urgency_score,
            punct_intensity,
            caps_ratio,
            negation_count,
            politeness,
            len(tokens)
        ])
    
    X_manual = np.array(features)
    X_combined = np.hstack([X_tfidf, X_manual])
    
    return X_combined, extractor


def train_models(save_dir=None):
    if save_dir is None:
        save_dir = SCRIPT_DIR
    else:
        save_dir = Path(save_dir)
    
    print("Generating synthetic email dataset...")
    df = generate_synthetic_dataset(600)
    df.to_csv(save_dir / 'emails_dataset.csv', index=False)
    print(f"Generated {len(df)} emails")
    
    print("\nExtracting features...")
    X, extractor = extract_email_features(df['email_text'])
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'vectorizer.pkl', 'wb') as f:
        pickle.dump(extractor, f)
    
    models = {}
    encoders = {}
    
    for target in ['category', 'urgency', 'tone']:
        print(f"\nTraining {target} models...")
        
        le = LabelEncoder()
        y = le.fit_transform(df[target])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_acc = lr.score(X_test, y_test)
        
        nb = MultinomialNB()
        nb.fit(X_train - X_train.min() + 1e-10, y_train)
        nb_acc = nb.score(X_test - X_test.min() + 1e-10, y_test)
        
        print(f"  LR accuracy: {lr_acc:.3f}")
        print(f"  NB accuracy: {nb_acc:.3f}")
        
        models[f'{target}_lr'] = lr
        models[f'{target}_nb'] = nb
        encoders[target] = le
        
        with open(save_dir / f'model_{target}_lr.pkl', 'wb') as f:
            pickle.dump(lr, f)
        with open(save_dir / f'model_{target}_nb.pkl', 'wb') as f:
            pickle.dump(nb, f)
        with open(save_dir / f'encoder_{target}.pkl', 'wb') as f:
            pickle.dump(le, f)
    
    print(f"\n✓ All models saved to {save_dir}/")
    return models, encoders, extractor


def load_models(model_dir=None):
    if model_dir is None:
        model_dir = SCRIPT_DIR
    else:
        model_dir = Path(model_dir)
    
    models = {}
    encoders = {}
    
    for target in ['category', 'urgency', 'tone']:
        with open(model_dir / f'model_{target}_lr.pkl', 'rb') as f:
            models[f'{target}_lr'] = pickle.load(f)
        with open(model_dir / f'model_{target}_nb.pkl', 'rb') as f:
            models[f'{target}_nb'] = pickle.load(f)
        with open(model_dir / f'encoder_{target}.pkl', 'rb') as f:
            encoders[target] = pickle.load(f)
    
    with open(model_dir / 'vectorizer.pkl', 'rb') as f:
        extractor = pickle.load(f)
    
    return models, encoders, extractor


def classify_email(email_text, models=None, encoders=None, extractor=None):
    if models is None:
        models, encoders, extractor = load_models()
    
    X_tfidf = extractor.transform_tfidf([email_text]).toarray()
    
    tokens = tokenize_text(clean_text(email_text))
    
    urgency_score = sum(
        1 for word in tokens 
        for keywords in URGENCY_KEYWORDS.values()
        if word.lower() in keywords
    )
    
    punct_intensity = email_text.count('!') + email_text.count('?') * 0.5
    caps_ratio = sum(1 for c in email_text if c.isupper()) / (len(email_text) + 1)
    negation_count = sum(1 for w in tokens if w.lower() in {'not', 'no', 'never'})
    politeness = sum(1 for w in tokens if w.lower() in {'please', 'kindly', 'thank'})
    
    X_manual = np.array([[urgency_score, punct_intensity, caps_ratio, negation_count, politeness, len(tokens)]])
    X = np.hstack([X_tfidf, X_manual])
    
    result = {
        'email': email_text,
        'predictions': {},
        'model_comparison': {},
        'explanations': {},
        'linguistic_features': {}
    }
    
    for target in ['category', 'urgency', 'tone']:
        lr_model = models[f'{target}_lr']
        nb_model = models[f'{target}_nb']
        le = encoders[target]
        
        lr_pred = lr_model.predict(X)[0]
        lr_proba = lr_model.predict_proba(X)[0]
        
        X_nb = X - X.min() + 1e-10
        nb_pred = nb_model.predict(X_nb)[0]
        nb_proba = nb_model.predict_proba(X_nb)[0]
        
        ensemble_proba = (lr_proba + nb_proba) / 2
        ensemble_pred = ensemble_proba.argmax()
        
        result['predictions'][target] = {
            'lr': le.inverse_transform([lr_pred])[0],
            'nb': le.inverse_transform([nb_pred])[0],
            'ensemble': le.inverse_transform([ensemble_pred])[0]
        }
        
        result['model_comparison'][target] = {
            'agreement': 100.0 if lr_pred == nb_pred else 0.0,
            'lr_confidence': float(lr_proba.max()),
            'nb_confidence': float(nb_proba.max())
        }
        
        if target == 'category':
            top_features = np.argsort(lr_model.coef_[lr_pred])[-5:]
            result['explanations']['top_features'] = top_features.tolist()
    
    tokens = tokenize_text(clean_text(email_text))
    result['linguistic_features'] = {
        'negation': sum(1 for w in tokens if w.lower() in {'not', 'no', 'never'}),
        'politeness': sum(1 for w in tokens if w.lower() in {'please', 'kindly', 'thank'}),
        'urgency_keywords': sum(1 for w in tokens if w.lower() in ['urgent', 'asap', 'immediately']),
        'punct_intensity': email_text.count('!') + email_text.count('?'),
        'length': len(tokens)
    }
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='IIIT Email Triage System')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--text', type=str, help='Email text to classify')
    parser.add_argument('--file', type=str, help='File with emails (one per line)')
    args = parser.parse_args()
    
    if args.train:
        train_models()
    elif args.text:
        result = classify_email(args.text)
        print(json.dumps(result, indent=2))
    elif args.file:
        with open(args.file) as f:
            emails = [line.strip() for line in f if line.strip()]
        models, encoders, extractor = load_models()
        for email in emails:
            result = classify_email(email, models, encoders, extractor)
            print(f"\n{'='*80}")
            print(f"Email: {email[:60]}...")
            print(f"Category: {result['predictions']['category']['ensemble']}")
            print(f"Urgency: {result['predictions']['urgency']['ensemble']}")
            print(f"Tone: {result['predictions']['tone']['ensemble']}")
    else:
        print("Usage: python triage.py [--train | --text '...' | --file emails.txt]")


if __name__ == '__main__':
    main()
