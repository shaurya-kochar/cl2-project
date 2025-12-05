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
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from preprocess import clean_text, tokenize_text
from features import FeatureExtractor


SENTIMENTS = ['Highly Positive', 'Positive', 'Neutral', 'Negative', 'Strongly Negative']

COMPLAINTS = [
    'Teaching Quality', 'TA Responsiveness', 'Course Difficulty',
    'Assignment Load', 'Evaluation Fairness', 'Lecture Pace',
    'Course Structure', 'Prerequisites', 'Infrastructure'
]

EMOTIONS = ['Confusion', 'Stress', 'Anger', 'Satisfaction', 'Appreciation', 'Overwhelm']

REVIEW_TEMPLATES = [
    ("The course was excellent! {prof} explained concepts very clearly.", 'Highly Positive', [], ['Satisfaction', 'Appreciation']),
    ("Loved this course. Assignments were challenging but fair.", 'Highly Positive', ['Assignment Load'], ['Satisfaction']),
    ("{prof}'s teaching style is amazing. Best course I've taken!", 'Highly Positive', ['Teaching Quality'], ['Appreciation']),
    
    ("Good course overall. Some topics were rushed though.", 'Positive', ['Lecture Pace'], ['Satisfaction']),
    ("Interesting content but assignments take too long.", 'Positive', ['Assignment Load'], ['Stress']),
    ("Decent course. TAs were helpful.", 'Positive', ['TA Responsiveness'], []),
    
    ("Average course. Nothing special.", 'Neutral', [], []),
    ("Course is okay. Could be better organized.", 'Neutral', ['Course Structure'], []),
    
    ("Too difficult for my level. Prerequisites weren't clear.", 'Negative', ['Course Difficulty', 'Prerequisites'], ['Confusion', 'Stress']),
    ("Assignments are impossible to complete. No help from TAs.", 'Negative', ['Assignment Load', 'TA Responsiveness'], ['Overwhelm', 'Anger']),
    ("Lectures are boring and hard to follow.", 'Negative', ['Teaching Quality'], ['Confusion']),
    
    ("Worst course ever! {prof} doesn't teach properly.", 'Strongly Negative', ['Teaching Quality'], ['Anger']),
    ("Completely unfair grading. Lost my CGPA because of this.", 'Strongly Negative', ['Evaluation Fairness'], ['Anger', 'Stress']),
    ("Terrible experience. Would NOT recommend.", 'Strongly Negative', [], ['Anger']),
]


def generate_course_reviews(n=800):
    data = []
    courses = ['COA', 'DS', 'DBMS', 'OS', 'AI', 'ML', 'CN', 'SE', 'Compilers', 'NLP']
    profs = ['Dr. Smith', 'Prof. Kumar', 'Dr. Patel', 'Prof. Sharma', 'Dr. Lee']
    
    for _ in range(n):
        template, sentiment, complaints, emotions = random.choice(REVIEW_TEMPLATES)
        
        review = template.format(
            prof=random.choice(profs),
            course=random.choice(courses)
        )
        
        data.append({
            'review_text': review,
            'sentiment': sentiment,
            'complaints': ','.join(complaints) if complaints else '',
            'emotions': ','.join(emotions) if emotions else ''
        })
    
    return pd.DataFrame(data)


def train_course_models(save_dir='apps/course_review'):
    print("Generating synthetic course reviews...")
    df = generate_course_reviews(800)
    df.to_csv(f'{save_dir}/reviews_dataset.csv', index=False)
    print(f"Generated {len(df)} reviews")
    
    print("\nExtracting features...")
    extractor = FeatureExtractor(tfidf_max_features=1000, tfidf_ngram=(1,2))
    extractor.fit_tfidf(df['review_text'])
    X = extractor.transform_tfidf(df['review_text']).toarray()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f'{save_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(extractor, f)
    
    print("\nTraining sentiment model...")
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(df['sentiment'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)
    
    lr_sentiment = LogisticRegression(max_iter=1000, random_state=42)
    lr_sentiment.fit(X_train, y_train)
    print(f"  LR accuracy: {lr_sentiment.score(X_test, y_test):.3f}")
    
    nb_sentiment = MultinomialNB()
    nb_sentiment.fit(X_train, y_train)
    print(f"  NB accuracy: {nb_sentiment.score(X_test, y_test):.3f}")
    
    with open(f'{save_dir}/model_sentiment_lr.pkl', 'wb') as f:
        pickle.dump(lr_sentiment, f)
    with open(f'{save_dir}/model_sentiment_nb.pkl', 'wb') as f:
        pickle.dump(nb_sentiment, f)
    with open(f'{save_dir}/encoder_sentiment.pkl', 'wb') as f:
        pickle.dump(le_sentiment, f)
    
    print("\nTraining emotion model (multi-label)...")
    mlb = MultiLabelBinarizer()
    y_emotions = mlb.fit_transform(df['emotions'].apply(lambda x: x.split(',') if x else []))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_emotions, test_size=0.2, random_state=42)
    
    lr_emotion = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    lr_emotion.fit(X_train, y_train)
    print(f"  LR accuracy: {lr_emotion.score(X_test, y_test):.3f}")
    
    with open(f'{save_dir}/model_emotion_lr.pkl', 'wb') as f:
        pickle.dump(lr_emotion, f)
    with open(f'{save_dir}/mlb_emotion.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    
    print(f"\n✓ All models saved to {save_dir}/")


def analyze_review(review_text, model_dir='apps/course_review'):
    with open(f'{model_dir}/vectorizer.pkl', 'rb') as f:
        extractor = pickle.load(f)
    with open(f'{model_dir}/model_sentiment_lr.pkl', 'rb') as f:
        lr_sentiment = pickle.load(f)
    with open(f'{model_dir}/model_sentiment_nb.pkl', 'rb') as f:
        nb_sentiment = pickle.load(f)
    with open(f'{model_dir}/encoder_sentiment.pkl', 'rb') as f:
        le_sentiment = pickle.load(f)
    with open(f'{model_dir}/model_emotion_lr.pkl', 'rb') as f:
        lr_emotion = pickle.load(f)
    with open(f'{model_dir}/mlb_emotion.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    X = extractor.transform_tfidf([review_text]).toarray()
    
    lr_pred = lr_sentiment.predict(X)[0]
    lr_proba = lr_sentiment.predict_proba(X)[0]
    nb_pred = nb_sentiment.predict(X)[0]
    nb_proba = nb_sentiment.predict_proba(X)[0]
    
    ensemble_proba = (lr_proba + nb_proba) / 2
    ensemble_pred = ensemble_proba.argmax()
    
    emotion_pred = lr_emotion.predict(X)
    emotions_detected = mlb.inverse_transform(emotion_pred)[0]
    
    result = {
        'review': review_text,
        'sentiment': {
            'lr': le_sentiment.inverse_transform([lr_pred])[0],
            'nb': le_sentiment.inverse_transform([nb_pred])[0],
            'ensemble': le_sentiment.inverse_transform([ensemble_pred])[0],
            'confidence': float(ensemble_proba.max())
        },
        'emotions': list(emotions_detected),
        'model_comparison': {
            'agreement': 100.0 if lr_pred == nb_pred else 0.0,
            'lr_confidence': float(lr_proba.max()),
            'nb_confidence': float(nb_proba.max())
        },
        'feature_importance': {
            'top_positive': np.argsort(lr_sentiment.coef_[ensemble_pred])[-5:].tolist(),
            'top_negative': np.argsort(lr_sentiment.coef_[ensemble_pred])[:5].tolist()
        }
    }
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='IIIT Course Review Analyzer')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--text', type=str, help='Review text to analyze')
    parser.add_argument('--file', type=str, help='File with reviews (one per line)')
    parser.add_argument('--app', action='store_true', help='Launch web app')
    args = parser.parse_args()
    
    if args.train:
        train_course_models()
    elif args.text:
        result = analyze_review(args.text)
        print(json.dumps(result, indent=2))
    elif args.file:
        with open(args.file) as f:
            reviews = [line.strip() for line in f if line.strip()]
        for review in reviews:
            result = analyze_review(review)
            print(f"\n{'='*80}")
            print(f"Review: {review[:60]}...")
            print(f"Sentiment: {result['sentiment']['ensemble']} ({result['sentiment']['confidence']:.2%})")
            print(f"Emotions: {', '.join(result['emotions']) if result['emotions'] else 'None'}")
    elif args.app:
        print("Launching Streamlit app...")
        import os
        os.system('streamlit run apps/course_review/app_ui.py')
    else:
        print("Usage: python review_analyzer.py [--train | --text '...' | --file reviews.txt | --app]")


if __name__ == '__main__':
    main()
