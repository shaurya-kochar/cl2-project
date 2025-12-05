# IIIT Email Triage System

Automated classification system for student emails to IIIT Hyderabad administrative offices.

## Features

- **Category Classification**: 11 categories (Attendance, Academic, Marks, Assignment, Hostel, Mess, Finance, Certificate, Medical, Technical, General)
- **Urgency Detection**: Critical, High, Medium, Low
- **Tone Analysis**: Angry, Frustrated, Confused, Polite, Neutral, Appreciative
- **Model Comparison**: LR vs NB with ensemble
- **Linguistic Features**: Negation, politeness, urgency keywords, punctuation intensity

## Quick Start

### Train Models

```bash
cd apps/email_triage
python triage.py --train
```

### Classify Single Email

```bash
python triage.py --text "Hi, my attendance for COA is not updated. Please check ASAP."
```

### Batch Processing

Create `emails.txt` with one email per line:

```bash
python triage.py --file emails.txt
```

## Example Output

```json
{
  "predictions": {
    "category": {"ensemble": "Attendance"},
    "urgency": {"ensemble": "High"},
    "tone": {"ensemble": "Polite"}
  },
  "model_comparison": {
    "category": {"agreement": 100.0}
  },
  "linguistic_features": {
    "negation": 1,
    "politeness": 2,
    "urgency_keywords": 1,
    "punct_intensity": 1
  }
}
```

## Models

- Logistic Regression (multi-class)
- Naive Bayes (multi-class)
- TF-IDF (unigrams + bigrams, 1000 features)
- Manual linguistic features (6 features)

## Dataset

600 synthetic IIIT-style emails generated automatically.
