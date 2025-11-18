# P20: Text Classification — Logistic Regression vs Naive Bayes

A research-grade implementation comparing classical ML approaches (Multinomial Naive Bayes and Logistic Regression) for sentiment classification on the IMDb movie review dataset. This project emphasizes feature engineering, interpretability, and robustness analysis without neural networks.

## Overview

This project implements and compares two classical text classification models with extensive feature engineering:
- **Multinomial Naive Bayes** 
- **Logistic Regression**

Key features include:
- TF-IDF with unigrams and bigrams
- POS-based features (noun/verb/adjective distributions)
- Negation and sentiment lexicon features
- Stylistic features (punctuation, sentence length, type-token ratio)
- Discourse markers and document structure signals
- Interpretability analysis (SHAP/LIME, feature coefficients)
- Adversarial robustness testing
- Per-phenomenon performance analysis

## Project Structure

```
cl2-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/
│   ├── README.md               # Dataset download instructions
│   ├── IMDB_Dataset.csv        # Raw dataset (not in git, see data/README.md)
│   └── processed.csv           # Preprocessed data (not in git)
├── src/
│   ├── preprocess.py           # Preprocessing pipeline
│   ├── features.py             # Feature engineering
│   ├── models.py               # Model implementations
│   ├── experiments.py          # Experiment runner
│   └── evaluate.py             # Evaluation and interpretability
├── notebooks/
│   ├── 01_explore.ipynb        # Data exploration
│   ├── 02_preprocess.ipynb     # Preprocessing demo
│   ├── 03_features.ipynb       # Feature engineering demo
│   ├── 04_models.ipynb         # Model training demo
│   └── 05_analysis.ipynb       # Results and interpretability
├── results/
│   ├── metrics.csv             # All experiment results
│   ├── experiment_matrix.csv   # Experiment configurations
│   ├── plots/                  # All figures for report
│   └── explanations.json       # LIME/SHAP interpretations
└── report/
    ├── report.tex              # LaTeX source
    └── report.pdf              # Final report

```

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
```

### 2. Download Dataset

Download the IMDb dataset and place it in `data/IMDB_Dataset.csv`. See `data/README.md` for detailed instructions.

Alternatively, use the HuggingFace snippet (in `data/README.md`) to download programmatically.

### 3. Run Baseline Experiment (E2: Logistic Regression with TF-IDF)

```bash
# Preprocess the data (quick mode with 5000 samples for testing)
python src/preprocess.py --input data/IMDB_Dataset.csv --output data/processed.csv --pos-limit 5000

# Run the baseline experiment
python src/experiments.py --matrix results/experiment_matrix.csv --out results/metrics.csv --run E2
```

For full reproduction (all 50k samples), remove `--pos-limit` flag. Full preprocessing takes ~20-30 minutes.

### Quick Test (2000 samples, ~5 minutes)

```bash
# Quick preprocessing
python src/preprocess.py --input data/IMDB_Dataset.csv --output data/processed_quick.csv --pos-limit 2000

# Run baseline on subset
python src/experiments.py --quick --matrix results/experiment_matrix.csv --out results/metrics_quick.csv
```

## Experiment Matrix

The project includes 7+ experiments comparing different feature combinations:

| Exp ID | Model | Features | Purpose |
|--------|-------|----------|---------|
| E1 | Naive Bayes | TF-IDF (unigram) | Baseline NB |
| E2 | Logistic Reg | TF-IDF (unigram) | Baseline LR |
| E3 | Naive Bayes | TF-IDF + POS | +Linguistic features |
| E4 | Logistic Reg | TF-IDF + POS | +Linguistic features |
| E5 | Naive Bayes | TF-IDF + Lexicon + Style | +Sentiment signals |
| E6 | Logistic Reg | TF-IDF + Lexicon + Style | +Sentiment signals |
| E7 | Ensemble | Average(NB, LR) | Hybrid approach |

Run all experiments:
```bash
python src/experiments.py --matrix results/experiment_matrix.csv --out results/metrics.csv
```

## Key Results

_(To be populated after experiments)_

Expected findings:
- Logistic Regression typically outperforms Naive Bayes by 2-3% accuracy
- Feature engineering beyond TF-IDF provides modest but consistent improvements
- Models struggle with negation and sarcasm
- Ensemble methods reduce variance

See `report/report.pdf` for detailed analysis and `notebooks/05_analysis.ipynb` for interactive exploration.

## Reproducibility

- All experiments use `RANDOM_SEED=42` for reproducibility
- Hardware tested: MacBook Air M1, 8GB RAM
- Full pipeline runtime: ~2-3 hours for all experiments
- Quick mode runtime: ~10-15 minutes

See `results/repro_instructions.txt` for detailed reproduction notes.

## Creative Extensions

This project includes several creative analyses beyond baseline classification:

1. **Hybrid Ensemble**: Probability averaging between NB and LR
2. **Windowed Negation Features**: Captures adjectives in negation scope
3. **Adversarial Testing**: Manual perturbations to test model fragility
4. **Phenomenon Partitions**: Performance on negation-heavy, hedging, long/short reviews
5. **Document Structure Signals**: First-sentence sentiment and discourse markers

## Report

The final report (`report/report.pdf`) includes:
- Literature review and methodology
- Detailed feature descriptions
- Experiment results with statistical significance tests
- Interpretability analysis (feature importance, LIME explanations)
- Adversarial case studies
- Limitations and future work

## License

MIT License - Academic/Research purposes

## Contact

Shaurya Kochar - [GitHub](https://github.com/shaurya-kochar)