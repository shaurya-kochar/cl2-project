#!/usr/bin/env python3

import sys
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from preprocess import clean_text

random.seed(42)
np.random.seed(42)

COURSES = [
    "Introduction to Programming", "Data Structures", "Algorithms",
    "Database Systems", "Machine Learning", "Computer Networks",
    "Operating Systems", "Software Engineering", "AI",
    "Computer Architecture", "Compiler Design", "Theory of Computation",
    "Computer Graphics", "Natural Language Processing", "Web Development"
]

PROFESSORS = [
    "Dr. Sharma", "Prof. Kumar", "Dr. Patel", "Prof. Singh", "Dr. Reddy",
    "Prof. Gupta", "Dr. Mehta", "Prof. Agarwal", "Dr. Rao", "Prof. Iyer"
]

POSITIVE_TEMPLATES = [
    "Excellent course! {prof} explains {topic} with great clarity. Highly recommend.",
    "Outstanding teaching by {prof}. The {topic} material is well-structured and engaging.",
    "{prof}'s lectures on {topic} are absolutely brilliant. Learned so much!",
    "Best course I've taken. {prof} makes {topic} concepts easy to understand.",
    "Amazing experience! {prof} is knowledgeable and passionate about {topic}.",
    "Fantastic course design. {prof}'s teaching style for {topic} is very effective.",
    "{prof} is an incredible instructor. The {topic} assignments are challenging but rewarding.",
    "Thoroughly enjoyed this course. {prof} brings {topic} to life with real examples.",
    "Superb lectures! {prof}'s approach to {topic} is innovative and practical.",
    "Exceptional course quality. {prof} goes above and beyond in teaching {topic}.",
]

NEUTRAL_TEMPLATES = [
    "The {topic} course by {prof} is okay. Content is standard, delivery is average.",
    "{prof}'s {topic} lectures are decent. Some topics are interesting, others less so.",
    "Satisfactory course on {topic}. {prof} covers the basics but nothing extraordinary.",
    "Average experience with {prof}'s {topic} class. Fulfills requirements.",
    "{topic} taught by {prof} is neither great nor bad. Just alright.",
    "The {topic} course is acceptable. {prof}'s teaching is competent but not inspiring.",
    "Fair course structure for {topic}. {prof} does an adequate job.",
    "{prof}'s {topic} class meets expectations. Nothing special to highlight.",
    "Standard course on {topic}. {prof}'s delivery is moderate.",
    "Reasonable course quality. {prof} handles {topic} in a straightforward manner.",
]

NEGATIVE_TEMPLATES = [
    "Disappointed with {prof}'s {topic} course. Lectures are unclear and confusing.",
    "Poor teaching quality. {prof} doesn't explain {topic} concepts well.",
    "Not satisfied with this {topic} course. {prof} seems unprepared often.",
    "Weak course design. {prof}'s approach to {topic} is ineffective.",
    "Frustrating experience. {prof} makes {topic} harder than it should be.",
    "Below expectations. {prof}'s {topic} lectures lack structure and clarity.",
    "Unimpressive course. {prof} struggles to communicate {topic} effectively.",
    "Difficult to follow. {prof}'s {topic} explanations are often vague.",
    "Not recommended. {prof}'s teaching style for {topic} doesn't work for me.",
    "Unsatisfactory course. {prof} needs to improve {topic} delivery significantly.",
]

HIGHLY_POSITIVE_TEMPLATES = [
    "Absolutely phenomenal! {prof} is a master at teaching {topic}. Life-changing course!",
    "Mind-blowing experience! {prof}'s {topic} lectures are world-class. Can't praise enough!",
    "Perfect 10/10! {prof} makes {topic} incredibly fascinating. Best professor ever!",
    "Revolutionary teaching! {prof}'s innovative methods for {topic} are spectacular!",
    "Simply outstanding! {prof} transforms complex {topic} into beautiful insights!",
    "Extraordinary course! {prof}'s passion for {topic} is contagious and inspiring!",
    "Unbelievably good! {prof} delivers {topic} content with unmatched expertise!",
    "Genius instruction! {prof}'s {topic} class exceeded all my expectations dramatically!",
    "Remarkable teaching! {prof} brings {topic} alive like no one else can!",
    "Truly exceptional! {prof}'s mastery of {topic} creates unforgettable learning!",
]

STRONGLY_NEGATIVE_TEMPLATES = [
    "Terrible course! {prof}'s {topic} teaching is completely inadequate and frustrating.",
    "Worst experience ever. {prof} has no idea how to teach {topic} properly.",
    "Absolutely awful! {prof}'s {topic} lectures are disorganized and incomprehensible.",
    "Complete disaster. {prof} makes {topic} unbearable with poor explanations.",
    "Extremely disappointed! {prof}'s approach to {topic} is fundamentally flawed.",
    "Horrible course design. {prof} fails to teach {topic} effectively at all.",
    "Dreadful experience. {prof}'s {topic} class is a waste of time and effort.",
    "Utterly terrible! {prof} cannot communicate {topic} concepts clearly whatsoever.",
    "Catastrophic teaching! {prof}'s {topic} course is disorganized and unhelpful.",
    "Abysmal quality! {prof}'s handling of {topic} is unacceptable and unprofessional.",
]

EMOTIONS_MAP = {
    "Highly Positive": ["Appreciation", "Satisfaction", "Excitement"],
    "Positive": ["Appreciation", "Satisfaction"],
    "Neutral": [],
    "Negative": ["Disappointment", "Frustration"],
    "Strongly Negative": ["Frustration", "Disappointment", "Anger"]
}


def generate_extended_dataset(n_samples=2000):
    data = []
    
    templates_map = {
        "Highly Positive": HIGHLY_POSITIVE_TEMPLATES,
        "Positive": POSITIVE_TEMPLATES,
        "Neutral": NEUTRAL_TEMPLATES,
        "Negative": NEGATIVE_TEMPLATES,
        "Strongly Negative": STRONGLY_NEGATIVE_TEMPLATES
    }
    
    samples_per_class = n_samples // 5
    
    for sentiment, templates in templates_map.items():
        for _ in range(samples_per_class):
            course = random.choice(COURSES)
            prof = random.choice(PROFESSORS)
            template = random.choice(templates)
            
            review = template.format(topic=course, prof=prof)
            
            emotions = EMOTIONS_MAP[sentiment].copy()
            if random.random() < 0.2 and emotions:
                emotions = random.sample(emotions, k=random.randint(1, len(emotions)))
            
            data.append({
                'review': review,
                'sentiment': sentiment,
                'emotions': ','.join(emotions),
                'course': course,
                'professor': prof
            })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def train_improved_models():
    print("Generating extended training dataset (2000 samples)...")
    df = generate_extended_dataset(n_samples=2000)
    
    output_dir = Path(__file__).parent
    df.to_csv(output_dir / 'reviews_dataset.csv', index=False)
    print(f"Dataset saved: {len(df)} reviews")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}\n")
    
    print("Preprocessing reviews...")
    df['review_clean'] = df['review'].apply(clean_text)
    
    X_train, X_test, y_sent_train, y_sent_test, y_emo_train, y_emo_test = train_test_split(
        df['review_clean'], df['sentiment'], df['emotions'],
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # TF-IDF with improved parameters
    print("\nExtracting TF-IDF features (5000 features, unigrams+bigrams)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Sentiment models with tuned parameters
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS")
    print("="*60)
    
    le_sentiment = LabelEncoder()
    y_sent_train_enc = le_sentiment.fit_transform(y_sent_train)
    y_sent_test_enc = le_sentiment.transform(y_sent_test)
    
    print("\nTraining Logistic Regression (C=10, max_iter=2000)...")
    lr_sentiment = LogisticRegression(C=10, max_iter=2000, random_state=42, n_jobs=-1)
    lr_sentiment.fit(X_train_tfidf, y_sent_train_enc)
    
    cv_scores_lr = cross_val_score(lr_sentiment, X_train_tfidf, y_sent_train_enc, cv=5)
    print(f"5-Fold CV Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")
    
    lr_pred = lr_sentiment.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_sent_test_enc, lr_pred)
    lr_f1 = f1_score(y_sent_test_enc, lr_pred, average='weighted')
    print(f"Test Accuracy: {lr_acc:.4f}")
    print(f"Test F1-Score: {lr_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_sent_test_enc, lr_pred, target_names=le_sentiment.classes_))
    
    print("\nTraining Naive Bayes (alpha=0.1)...")
    nb_sentiment = MultinomialNB(alpha=0.1)
    nb_sentiment.fit(X_train_tfidf, y_sent_train_enc)
    
    cv_scores_nb = cross_val_score(nb_sentiment, X_train_tfidf, y_sent_train_enc, cv=5)
    print(f"5-Fold CV Accuracy: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std():.4f})")
    
    nb_pred = nb_sentiment.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_sent_test_enc, nb_pred)
    nb_f1 = f1_score(y_sent_test_enc, nb_pred, average='weighted')
    print(f"Test Accuracy: {nb_acc:.4f}")
    print(f"Test F1-Score: {nb_f1:.4f}")
    
    # Emotion models
    print("\n" + "="*60)
    print("EMOTION DETECTION")
    print("="*60)
    
    y_emo_train_list = [e.split(',') if e else [] for e in y_emo_train]
    y_emo_test_list = [e.split(',') if e else [] for e in y_emo_test]
    
    mlb = MultiLabelBinarizer()
    y_emo_train_bin = mlb.fit_transform(y_emo_train_list)
    y_emo_test_bin = mlb.transform(y_emo_test_list)
    
    print(f"\nEmotion classes: {list(mlb.classes_)}")
    print(f"Multi-label matrix shape: {y_emo_train_bin.shape}")
    
    print("\nTraining Logistic Regression (C=5, max_iter=2000)...")
    from sklearn.multioutput import MultiOutputClassifier
    lr_emotion = MultiOutputClassifier(
        LogisticRegression(C=5, max_iter=2000, random_state=42, n_jobs=-1)
    )
    lr_emotion.fit(X_train_tfidf, y_emo_train_bin)
    
    lr_emo_pred = lr_emotion.predict(X_test_tfidf)
    lr_emo_acc = accuracy_score(y_emo_test_bin, lr_emo_pred)
    lr_emo_f1 = f1_score(y_emo_test_bin, lr_emo_pred, average='weighted', zero_division=0)
    print(f"Test Subset Accuracy: {lr_emo_acc:.4f}")
    print(f"Test F1-Score: {lr_emo_f1:.4f}")
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    with open(output_dir / 'vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(output_dir / 'model_sentiment_lr.pkl', 'wb') as f:
        pickle.dump(lr_sentiment, f)
    with open(output_dir / 'model_sentiment_nb.pkl', 'wb') as f:
        pickle.dump(nb_sentiment, f)
    with open(output_dir / 'encoder_sentiment.pkl', 'wb') as f:
        pickle.dump(le_sentiment, f)
    with open(output_dir / 'model_emotion_lr.pkl', 'wb') as f:
        pickle.dump(lr_emotion, f)
    with open(output_dir / 'mlb_emotion.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    
    print("\nAll models saved successfully!")
    print(f"  - vectorizer.pkl")
    print(f"  - model_sentiment_lr.pkl (Acc: {lr_acc:.4f}, F1: {lr_f1:.4f})")
    print(f"  - model_sentiment_nb.pkl (Acc: {nb_acc:.4f}, F1: {nb_f1:.4f})")
    print(f"  - encoder_sentiment.pkl")
    print(f"  - model_emotion_lr.pkl (Acc: {lr_emo_acc:.4f}, F1: {lr_emo_f1:.4f})")
    print(f"  - mlb_emotion.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Dataset: {len(df)} reviews (5 sentiment classes, {len(mlb.classes_)} emotions)")
    print(f"  Best Sentiment Model: {'LR' if lr_acc > nb_acc else 'NB'} ({max(lr_acc, nb_acc):.4f})")
    print(f"  Emotion Model: LR ({lr_emo_acc:.4f})")


if __name__ == '__main__':
    train_improved_models()
