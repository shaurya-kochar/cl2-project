# P20: Text Classification - Complete Evaluation Report

## Executive Summary

This report presents a comprehensive evaluation of multiple machine learning models
for sentiment classification of IMDB movie reviews. The evaluation covers 8 different
classification algorithms with extensive metrics and visualizations.

**Best Performing Model:** Logistic Regression
- Accuracy: 0.9600
- F1 Score: 0.9600
- MCC: 0.9200

---

## 1. Dataset Overview

- **Dataset Size:** 3,000 movie reviews
- **Class Distribution:** Balanced (~50% positive, ~50% negative)
- **Features:** TF-IDF vectorization with unigrams and bigrams
- **Feature Dimension:** ~2,000 features

---

## 2. Models Evaluated

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier with L2 regularization |
| Naive Bayes | Multinomial NB for text classification |
| SVM (Linear) | Linear Support Vector Machine |
| Random Forest | Ensemble of 100 decision trees |
| Gradient Boosting | Sequential ensemble boosting |
| Decision Tree | Single CART tree classifier |
| KNN | K-Nearest Neighbors (k=5) |
| AdaBoost | Adaptive Boosting ensemble |

---

## 3. Performance Metrics

### 3.1 Test Set Results

| Model | Accuracy | Precision | Recall | F1 | MCC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.9600 | 0.9600 | 0.9600 | 0.9600 | 0.9200 |
| Naive Bayes | 0.9600 | 0.9600 | 0.9600 | 0.9600 | 0.9200 |
| SVM (Linear) | 0.9583 | 0.9583 | 0.9583 | 0.9583 | 0.9166 |
| Random Forest | 0.9133 | 0.9180 | 0.9133 | 0.9130 | 0.8311 |
| Gradient Boosting | 0.9467 | 0.9467 | 0.9467 | 0.9467 | 0.8934 |
| Decision Tree | 0.8917 | 0.9013 | 0.8917 | 0.8909 | 0.7925 |
| KNN | 0.9500 | 0.9501 | 0.9500 | 0.9500 | 0.9000 |
| AdaBoost | 0.8417 | 0.8624 | 0.8417 | 0.8397 | 0.7044 |

### 3.2 Cross-Validation Results (5-Fold)

| Model | CV Mean | CV Std |
|-------|---------|--------|
| Logistic Regression | 0.9510 | 0.0084 |
| Naive Bayes | 0.9537 | 0.0078 |
| SVM (Linear) | 0.9523 | 0.0062 |
| Random Forest | 0.9013 | 0.0172 |
| Gradient Boosting | 0.9403 | 0.0059 |
| Decision Tree | 0.8817 | 0.0089 |
| KNN | 0.9433 | 0.0071 |
| AdaBoost | 0.8497 | 0.0054 |

---

## 4. Key Findings

### 4.1 Model Performance Analysis

1. **Logistic Regression** and **Naive Bayes** achieved the highest accuracy (0.9600),
   demonstrating that linear models work well for text classification.

2. **SVM (Linear)** showed comparable performance, confirming that linear decision
   boundaries are effective for sentiment analysis.

3. **Tree-based models** (Random Forest, Decision Tree) showed slightly lower performance,
   possibly due to the high-dimensional sparse nature of text data.

4. **AdaBoost** had the lowest performance, suggesting that boosting on weak learners
   may not be optimal for this text classification task.

### 4.2 Feature Importance

The most discriminative features for classification were:
- Positive: "amazing", "excellent", "love", "best", "great"
- Negative: "terrible", "boring", "waste", "bad", "avoid"

### 4.3 Error Analysis

Common misclassification patterns:
- Reviews with mixed sentiment signals
- Short reviews with limited context
- Sarcastic or ironic reviews

---

## 5. Visualizations Generated

1. **data_distribution.png** - Dataset statistics and class distribution
2. **model_comparison.png** - Bar charts comparing all metrics across models
3. **confusion_matrices.png** - Confusion matrices for each classifier
4. **roc_curves.png** - ROC curves with AUC scores
5. **precision_recall_curves.png** - Precision-recall curves
6. **learning_curves.png** - Learning curves showing sample efficiency
7. **cross_validation.png** - Cross-validation box plots
8. **feature_importance.png** - Top features by importance
9. **metrics_heatmap.png** - Heatmap of all metrics
10. **calibration_curves.png** - Probability calibration analysis
11. **error_analysis.png** - Detailed error breakdown
12. **summary_dashboard.png** - Comprehensive overview dashboard
13. **word_clouds.png** - Word clouds for each sentiment
14. **word_frequency.png** - Word frequency analysis
15. **radar_chart.png** - Radar chart model comparison
16. **training_dynamics.png** - Training progress simulation

---

## 6. Conclusions

1. **Linear models are highly effective** for sentiment classification,
   with Logistic Regression and Naive Bayes achieving the best results.

2. **TF-IDF features with n-grams** provide strong signal for classification.

3. **Cross-validation results are consistent** with test set performance,
   indicating good generalization.

4. **All models exceed 84% accuracy**, showing the task is well-suited
   to supervised learning approaches.

---

## 7. Recommendations

1. Deploy Logistic Regression for production due to its:
   - High accuracy and interpretability
   - Fast inference time
   - Robust calibrated probabilities

2. Consider ensemble methods combining top performers for marginal improvement.

3. Explore deep learning (BERT, transformers) for potential further gains.

---

*Report generated automatically by P20 evaluation pipeline*
