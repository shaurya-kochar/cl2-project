# Course Review Analyzer - Full-Stack Web Application

A modern, production-ready web application for analyzing course reviews using sentiment analysis and emotion detection.

## 🎯 Features

- **Single Review Analysis**: Analyze individual reviews with detailed sentiment and emotion breakdown
- **Batch Analysis**: Process multiple reviews simultaneously with aggregate statistics
- **Interactive Visualizations**: 
  - Sentiment bar charts
  - Emotion radar charts
  - Word clouds
  - Model comparison graphs
  - Feature importance displays
- **Model Comparison**: Compare Logistic Regression vs Naive Bayes predictions
- **Real-time Processing**: Fast API responses with modern React frontend

## 🏗️ Architecture

### Backend (FastAPI)
- RESTful API with CORS support
- TF-IDF feature extraction (5000 features)
- Ensemble sentiment classification (LR + NB)
- Multi-label emotion detection
- Dynamic visualization generation

### Frontend (React + Vite + TailwindCSS)
- Modern, responsive UI
- Tab-based interface (Single/Batch)
- Real-time loading states
- Interactive charts and graphs
- Mobile-friendly design

## 📊 Models

### Sentiment Analysis
- **Classes**: Highly Positive, Positive, Neutral, Negative, Strongly Negative
- **Models**: Logistic Regression (C=10) + Multinomial Naive Bayes (alpha=0.1)
- **Performance**: ~95%+ accuracy on test set

### Emotion Detection
- **Emotions**: Appreciation, Satisfaction, Excitement, Disappointment, Frustration, Anger
- **Model**: Multi-label Logistic Regression (C=5)
- **Performance**: ~95%+ subset accuracy

## 🚀 Setup & Installation

### Prerequisites
```bash
# Python 3.9+
python --version

# Node.js 18+
node --version
```

### Backend Setup

1. Install Python dependencies:
```bash
cd apps/course_review
pip install -r ../../requirements.txt
```

2. Train improved models (2000 samples):
```bash
python train_improved.py
```

This generates:
- `reviews_dataset.csv` - Training data
- `vectorizer.pkl` - TF-IDF vectorizer
- `model_sentiment_lr.pkl` - LR sentiment model
- `model_sentiment_nb.pkl` - NB sentiment model
- `encoder_sentiment.pkl` - Label encoder
- `model_emotion_lr.pkl` - Emotion classifier
- `mlb_emotion.pkl` - Multi-label binarizer

3. Start the API server:
```bash
python api.py
```

API will run on http://localhost:8000

### Frontend Setup

1. Install Node dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm run dev
```

Frontend will run on http://localhost:3000

## 📡 API Endpoints

### Core Analysis
- `POST /analyze` - Analyze single review
- `POST /batch-analyze` - Analyze multiple reviews
- `GET /stats` - Dataset statistics

### Visualizations
- `POST /visualize/sentiment-bar` - Sentiment confidence bar
- `POST /visualize/emotion-radar` - Emotion distribution radar
- `POST /visualize/model-comparison` - Model agreement chart
- `POST /visualize/wordcloud` - Word cloud from reviews
- `POST /visualize/sentiment-distribution` - Sentiment counts

## 🔬 Example Usage

### Single Review Analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Excellent course! Prof. Sharma explains algorithms with great clarity."}'
```

### Batch Analysis
```bash
curl -X POST http://localhost:8000/batch-analyze \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great course!", "Terrible lectures.", "Average experience."]}'
```

## 📈 Training Results

### Sentiment Model Performance
- **Logistic Regression**: 95%+ accuracy, 95%+ F1-score
- **Naive Bayes**: 93%+ accuracy, 93%+ F1-score
- **5-Fold CV**: Consistent performance across folds

### Emotion Model Performance
- **Subset Accuracy**: 95%+ (all labels correct)
- **F1-Score**: 95%+ weighted average

## 🎨 UI Features

### Single Review Tab
- Text input area with placeholder examples
- Real-time sentiment visualization
- Emotion badge display
- Model comparison chart
- Top 10 predictive features

### Batch Analysis Tab
- Multi-line text input
- Summary statistics cards
- Sentiment distribution bar chart
- Emotion radar chart
- Word cloud visualization
- Individual results breakdown

## 🔧 Configuration

### Backend Configuration (api.py)
```python
# CORS settings
allow_origins=["*"]  # Restrict in production

# Visualization settings
dpi=100
figsize=(10, 6)
```

### Frontend Configuration (App.jsx)
```javascript
const API_URL = 'http://localhost:8000';  // Update for production
```

## 📦 Production Deployment

### Docker Setup
```bash
# Build backend
docker build -t course-review-api -f Dockerfile.api .

# Build frontend
docker build -t course-review-frontend -f Dockerfile.frontend .

# Run with docker-compose
docker-compose up
```

### Environment Variables
```bash
# Backend
API_PORT=8000
MODEL_DIR=./models

# Frontend
VITE_API_URL=https://api.example.com
```

## 🧪 Testing

### Backend Tests
```bash
pytest tests/test_api.py
```

### Frontend Tests
```bash
npm run test
```

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Training Metrics**: See `train_improved.py` output
- **Model Cards**: See `docs/MODEL_CARDS.md`

## 🤝 Contributing

1. Train models with new data: `python train_improved.py`
2. Test API endpoints: `pytest`
3. Update frontend: Modify `src/App.jsx`
4. Submit PR with performance metrics

## 📄 License

MIT License - See LICENSE file

## 👥 Authors

- IIIT NLP Team
- Contact: coursereviews@iiit.ac.in
