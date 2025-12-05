#!/usr/bin/env python3
"""
IIIT CourseReview Analyzer - Complete FastAPI Backend
All functionality in one file for reliability
"""

import io
import os
import re
import base64
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from wordcloud import WordCloud

# Get the directory where this file is located
MODEL_DIR = Path(__file__).parent.resolve()

app = FastAPI(title="IIIT CourseReview Analyzer API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for models
models_cache = {}


# ===================== DATA MODELS =====================

class ReviewRequest(BaseModel):
    text: str


class BatchReviewRequest(BaseModel):
    reviews: List[str]


# ===================== HELPER FUNCTIONS =====================

def clean_text(text: str) -> str:
    """Simple text cleaning"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def load_models():
    """Load all trained models into cache"""
    global models_cache
    
    if models_cache:
        return models_cache
    
    print(f"Loading models from: {MODEL_DIR}")
    
    try:
        with open(MODEL_DIR / 'vectorizer.pkl', 'rb') as f:
            models_cache['vectorizer'] = pickle.load(f)
        with open(MODEL_DIR / 'model_sentiment_lr.pkl', 'rb') as f:
            models_cache['lr_sentiment'] = pickle.load(f)
        with open(MODEL_DIR / 'model_sentiment_nb.pkl', 'rb') as f:
            models_cache['nb_sentiment'] = pickle.load(f)
        with open(MODEL_DIR / 'encoder_sentiment.pkl', 'rb') as f:
            models_cache['le_sentiment'] = pickle.load(f)
        with open(MODEL_DIR / 'model_emotion_lr.pkl', 'rb') as f:
            models_cache['lr_emotion'] = pickle.load(f)
        with open(MODEL_DIR / 'mlb_emotion.pkl', 'rb') as f:
            models_cache['mlb'] = pickle.load(f)
        
        print(f"✓ All models loaded successfully")
        return models_cache
    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}")


def analyze_single_review(text: str) -> dict:
    """Analyze a single review and return detailed results"""
    models = load_models()
    
    vectorizer = models['vectorizer']
    lr_sentiment = models['lr_sentiment']
    nb_sentiment = models['nb_sentiment']
    le_sentiment = models['le_sentiment']
    lr_emotion = models['lr_emotion']
    mlb = models['mlb']
    
    # Clean and vectorize
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    
    # Sentiment predictions
    lr_pred = lr_sentiment.predict(X)[0]
    lr_proba = lr_sentiment.predict_proba(X)[0]
    nb_pred = nb_sentiment.predict(X)[0]
    nb_proba = nb_sentiment.predict_proba(X)[0]
    
    # Ensemble prediction
    ensemble_proba = (lr_proba + nb_proba) / 2
    ensemble_pred = ensemble_proba.argmax()
    
    # Emotion predictions
    emotion_pred = lr_emotion.predict(X)
    emotions_detected = list(mlb.inverse_transform(emotion_pred)[0])
    
    # Get feature names and importance
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top features for this prediction
    try:
        coef = lr_sentiment.coef_[ensemble_pred]
        top_indices = np.argsort(np.abs(coef))[-10:][::-1]
        feature_importance = {feature_names[i]: float(np.abs(coef[i])) for i in top_indices}
        # Normalize
        max_imp = max(feature_importance.values()) if feature_importance else 1
        feature_importance = {k: v/max_imp for k, v in feature_importance.items()}
    except Exception:
        feature_importance = {}
    
    return {
        'review': text,
        'sentiment': {
            'lr': le_sentiment.inverse_transform([lr_pred])[0],
            'nb': le_sentiment.inverse_transform([nb_pred])[0],
            'ensemble': le_sentiment.inverse_transform([ensemble_pred])[0],
            'confidence': float(ensemble_proba.max())
        },
        'emotions': emotions_detected,
        'model_comparison': {
            'agreement': 100.0 if lr_pred == nb_pred else 0.0,
            'lr_confidence': float(lr_proba.max()),
            'nb_confidence': float(nb_proba.max())
        },
        'feature_importance': feature_importance
    }


# ===================== API ENDPOINTS =====================

@app.get("/")
def root():
    """API health check"""
    return {
        "message": "IIIT CourseReview Analyzer API",
        "status": "running",
        "version": "2.0"
    }


@app.get("/health")
def health_check():
    """Check if models are loaded"""
    try:
        load_models()
        return {"status": "healthy", "models_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/analyze")
def analyze(request: ReviewRequest):
    """Analyze a single review"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        result = analyze_single_review(request.text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-analyze")
def batch_analyze(request: BatchReviewRequest):
    """Analyze multiple reviews"""
    try:
        reviews = [r.strip() for r in request.reviews if r.strip()]
        if not reviews:
            raise HTTPException(status_code=400, detail="No valid reviews provided")
        
        results = [analyze_single_review(review) for review in reviews]
        
        # Calculate summary statistics
        sentiments = [r['sentiment']['ensemble'] for r in results]
        sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
        
        all_emotions = []
        for r in results:
            all_emotions.extend(r['emotions'])
        emotion_counts = pd.Series(all_emotions).value_counts().to_dict() if all_emotions else {}
        
        avg_confidence = np.mean([r['sentiment']['confidence'] for r in results])
        avg_agreement = np.mean([r['model_comparison']['agreement'] for r in results])
        
        return {
            "results": results,
            "summary": {
                "total_reviews": len(results),
                "sentiment_distribution": sentiment_counts,
                "emotion_distribution": emotion_counts,
                "avg_confidence": float(avg_confidence),
                "avg_model_agreement": float(avg_agreement)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/sentiment-gauge")
def visualize_sentiment_gauge(request: ReviewRequest):
    """Create a sentiment gauge visualization"""
    try:
        result = analyze_single_review(request.text)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        
        sentiments = ['Strongly\nNegative', 'Negative', 'Neutral', 'Positive', 'Highly\nPositive']
        colors = ['#ef4444', '#f97316', '#fbbf24', '#84cc16', '#10b981']
        
        # Map result to index
        sentiment_map = {
            'Strongly Negative': 0, 'Negative': 1, 'Neutral': 2, 
            'Positive': 3, 'Highly Positive': 4
        }
        current_idx = sentiment_map.get(result['sentiment']['ensemble'], 2)
        
        for i, (sent, color) in enumerate(zip(sentiments, colors)):
            alpha = 1.0 if i == current_idx else 0.3
            ax.barh(0, 1, left=i, color=color, alpha=alpha, edgecolor='white', linewidth=3)
            ax.text(i + 0.5, 0, sent, ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white' if i == current_idx else 'gray')
        
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        conf = result['sentiment']['confidence']
        ax.set_title(f"Sentiment: {result['sentiment']['ensemble']} (Confidence: {conf:.1%})", 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return {"image": fig_to_base64(fig), "sentiment": result['sentiment']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/sentiment-bar")
def visualize_sentiment_bar(request: ReviewRequest):
    """Create sentiment bar visualization"""
    try:
        result = analyze_single_review(request.text)
        
        fig, ax = plt.subplots(figsize=(10, 2.5))
        
        sentiments = ['Highly Positive', 'Positive', 'Neutral', 'Negative', 'Strongly Negative']
        colors = ['#10b981', '#84cc16', '#fbbf24', '#f97316', '#ef4444']
        
        # Find the current sentiment
        current_sentiment = result['sentiment']['ensemble']
        try:
            sentiment_idx = sentiments.index(current_sentiment)
        except ValueError:
            sentiment_idx = 2  # Default to Neutral
        
        for i, (sent, color) in enumerate(zip(sentiments, colors)):
            alpha = 1.0 if i == sentiment_idx else 0.3
            ax.barh(0, 1, left=i, color=color, alpha=alpha, edgecolor='white', linewidth=2)
            short_name = sent.split()[0]
            ax.text(i + 0.5, 0, short_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white' if i == sentiment_idx else 'gray')
        
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.set_title(f"Sentiment: {current_sentiment} ({result['sentiment']['confidence']:.1%})", 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return {"image": fig_to_base64(fig)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/model-comparison")
def visualize_model_comparison(request: ReviewRequest):
    """Create model comparison bar chart"""
    try:
        result = analyze_single_review(request.text)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        models = ['Logistic\nRegression', 'Naive\nBayes', 'Ensemble']
        confidences = [
            result['model_comparison']['lr_confidence'],
            result['model_comparison']['nb_confidence'],
            result['sentiment']['confidence']
        ]
        colors = ['#3b82f6', '#8b5cf6', '#10b981']
        
        bars = ax.bar(models, confidences, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{conf:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Confidence Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add agreement indicator
        agreement = result['model_comparison']['agreement']
        ax.text(0.5, -0.12, f"Model Agreement: {agreement:.0f}%", 
               transform=ax.transAxes, ha='center', fontsize=11, 
               color='#059669' if agreement == 100 else '#dc2626')
        
        plt.tight_layout()
        return {"image": fig_to_base64(fig), "comparison": result['model_comparison']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/emotion-radar")
def visualize_emotion_radar(request: BatchReviewRequest):
    """Create emotion radar chart"""
    try:
        results = [analyze_single_review(r) for r in request.reviews if r.strip()]
        
        all_emotions = []
        for r in results:
            all_emotions.extend(r['emotions'])
        
        if not all_emotions:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No emotions detected', ha='center', va='center', 
                   fontsize=14, color='#6b7280', transform=ax.transAxes)
            ax.axis('off')
            plt.tight_layout()
            return {"image": fig_to_base64(fig)}
        
        emotion_counts = pd.Series(all_emotions).value_counts()
        
        categories = list(emotion_counts.index)
        values = list(emotion_counts.values)
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#6366f1')
        ax.fill(angles_plot, values_plot, alpha=0.25, color='#6366f1')
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title("Emotion Distribution", fontsize=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return {"image": fig_to_base64(fig)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/wordcloud")
def visualize_wordcloud(request: BatchReviewRequest):
    """Create word cloud from reviews"""
    try:
        text = ' '.join(request.reviews)
        cleaned = clean_text(text)
        
        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="No valid text for word cloud")
        
        wordcloud = WordCloud(
            width=900, height=450, 
            background_color='white',
            colormap='viridis', 
            max_words=80
        ).generate(cleaned)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Review Word Cloud', fontsize=16, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return {"image": fig_to_base64(fig)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/sentiment-distribution")
def visualize_sentiment_distribution(request: BatchReviewRequest):
    """Create sentiment distribution chart"""
    try:
        results = [analyze_single_review(r) for r in request.reviews if r.strip()]
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid reviews to analyze")
        
        sentiments = [r['sentiment']['ensemble'] for r in results]
        sentiment_order = ['Strongly Negative', 'Negative', 'Neutral', 'Positive', 'Highly Positive']
        
        counts = pd.Series(sentiments).value_counts()
        counts = counts.reindex(sentiment_order, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#ef4444', '#f97316', '#fbbf24', '#84cc16', '#10b981']
        
        bars = ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.85, 
                     edgecolor='white', linewidth=2)
        
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([s.replace(' ', '\n') for s in counts.index], fontsize=10)
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Sentiment Distribution ({len(results)} reviews)', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        for bar, count in zip(bars, counts.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        return {"image": fig_to_base64(fig), "distribution": counts.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Get dataset statistics"""
    try:
        dataset_path = MODEL_DIR / 'reviews_dataset.csv'
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            
            sentiment_dist = df['sentiment'].value_counts().to_dict()
            
            emotion_counts = {}
            for emotions_str in df['emotions']:
                if emotions_str and isinstance(emotions_str, str):
                    for emotion in emotions_str.split(','):
                        emotion = emotion.strip()
                        if emotion:
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            return {
                "total_reviews": len(df),
                "sentiment_distribution": sentiment_dist,
                "emotion_distribution": emotion_counts
            }
        else:
            return {"total_reviews": 0, "message": "No dataset found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== STARTUP =====================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_models()
        print("✓ API ready - Models loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load models - {e}")
        print("  Run training first: python train_improved.py")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting API server...")
    print(f"Model directory: {MODEL_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
