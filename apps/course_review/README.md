# IIIT CourseReview Analyzer

Web-based sentiment and emotion analysis for student course feedback.

## Features

- **5-Level Sentiment**: Highly Positive → Strongly Negative
- **Multi-label Emotions**: Confusion, Stress, Anger, Satisfaction, Appreciation, Overwhelm
- **Model Comparison**: LR vs NB with ensemble
- **Interactive UI**: Real-time analysis with Streamlit
- **Explainability**: Feature importance visualization

## Quick Start

### Train Models

```bash
cd apps/course_review
python review_analyzer.py --train
```

### Analyze Single Review

```bash
python review_analyzer.py --text "The course was excellent! Best class ever."
```

### Launch Web App

```bash
python review_analyzer.py --app
```

Or directly:

```bash
streamlit run app_ui.py
```

## Web Interface

The Streamlit app provides:

- ✅ Sentiment prediction with confidence
- 😊 Emotion detection (multi-label)
- 📊 Model agreement metrics
- 🔍 LR vs NB comparison
- 💡 Feature importance display

## Example Output

```json
{
  "sentiment": {
    "ensemble": "Highly Positive",
    "confidence": 0.98
  },
  "emotions": ["Satisfaction", "Appreciation"],
  "model_comparison": {
    "agreement": 100.0,
    "lr_confidence": 0.97,
    "nb_confidence": 0.99
  }
}
```

## Models

- Logistic Regression (multi-class sentiment, multi-label emotions)
- Naive Bayes (multi-class sentiment)
- TF-IDF vectorization (1000 features, unigrams + bigrams)

## Dataset

800 synthetic IIIT course reviews across 5 sentiment levels.
