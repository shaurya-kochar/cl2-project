# Extension Features Documentation

This project includes advanced linguistic features inspired by multiple NLP components beyond basic text classification.

## Overview

The `src/extensions.py` module provides **28 additional features** across 4 categories:
- P13: Enhanced Sentiment Analysis (9 features)
- Anaphora Resolution (8 features)  
- RST-inspired Discourse Analysis (11 features)
- P14: Semantic Density (3 features, placeholder for embeddings)

Total: **28 extension features** + **18 baseline features** = **46 linguistic features**

## Feature Categories

### 1. P13: Enhanced Sentiment Analysis (9 features)

**Emotion Detection** - Multi-class emotion scoring:
- `emotion_joy`: Count of joy-related words (happy, delight, wonderful, love, etc.)
- `emotion_anger`: Count of anger words (angry, hate, terrible, awful, etc.)
- `emotion_sadness`: Count of sadness words (sad, disappointed, tragic, etc.)
- `emotion_fear`: Count of fear words (afraid, scared, worried, etc.)
- `emotion_surprise`: Count of surprise words (shocked, amazed, unexpected, etc.)

**Polarity Analysis**:
- `polarity_positive`: Count of positive sentiment words
- `polarity_negative`: Count of negative sentiment words
- `polarity_ratio`: Ratio of positive to negative words
- `intensity_modifiers`: Count of intensity modifiers (very, extremely, absolutely, etc.)

### 2. Anaphora Resolution Features (8 features)

**Pronoun Analysis** - Counts by pronoun type:
- `pronoun_personal`: I, me, my, we, us, you, your, etc.
- `pronoun_demonstrative`: this, that, these, those
- `pronoun_relative`: who, whom, whose, which
- `pronoun_reflexive`: myself, yourself, himself, etc.

**Contextual Features**:
- `pronoun_density`: Ratio of pronouns to total tokens
- `pronoun_noun_distance`: Average distance between pronouns and nearest nouns (anaphora span)

### 3. RST-inspired Discourse Salience (11 features)

**Sentence Position Analysis**:
- `first_sent_salience`: Positional weight of first sentence (1.0 - normalized position)
- `last_sent_salience`: Positional weight of last sentence
- `max_sentiment_position`: Position of sentence with highest sentiment score
- `sent_length_variance`: Variance in sentence lengths (style consistency)
- `num_sentences`: Total sentence count

**Discourse Connectives** - Counts by relation type:
- `discourse_contrast`: but, however, although, though, yet, nevertheless, etc.
- `discourse_cause`: because, since, therefore, thus, hence, etc.
- `discourse_addition`: moreover, furthermore, additionally, also, etc.
- `discourse_temporal`: then, meanwhile, afterwards, subsequently, etc.
- `discourse_example`: for example, for instance, such as, etc.

### 4. P14: Semantic Density Features (3 features)

**Content Word Analysis**:
- `content_word_density`: Ratio of content words (nouns, verbs, adjectives, adverbs) to total tokens
- `content_word_diversity`: Unique content words / total content words (lexical richness)
- `function_word_ratio`: Ratio of function words (determiners, prepositions, etc.)

**Note**: Full semantic embeddings (BERT/GloVe) are not included due to computational constraints. These density features provide a lightweight alternative measuring semantic richness.

## Usage

### Standalone Extension Features

```bash
python src/extensions.py --input data/processed.csv --output results/extension_features.csv
```

### Integrated with Baseline Features

```bash
python src/features.py --input data/processed.csv --output results/features_all.csv --extensions
```

The `--extensions` flag adds all 28 extension features to the baseline 18 features.

### In Python

```python
from extensions import extract_all_extension_features
import pandas as pd

df = pd.read_csv('data/processed.csv')
extensions = extract_all_extension_features(df)
print(extensions.shape)  # (n_samples, 28)
```

## Integration with Classification

Extension features can improve model performance by:

1. **Emotion-aware classification**: Multi-emotion scores capture nuanced sentiment beyond binary positive/negative
2. **Contextual understanding**: Pronoun density and anaphora distance indicate narrative coherence
3. **Discourse structure**: Connective usage reveals argumentative vs. narrative writing styles
4. **Semantic richness**: Content word density correlates with review informativeness

### Experiment Configurations

Add extension experiments to the matrix:

| ID  | Model | Features | Purpose |
|-----|-------|----------|---------|
| E8  | LR    | TF-IDF + Extensions | Test emotion features |
| E9  | NB    | TF-IDF + Extensions | Test discourse features |
| E10 | LR    | TF-IDF + POS + Extensions | Full feature set |

## Cross-Component Connections

### P13: Sentiment Analysis
- **Emotion lexicons** provide fine-grained sentiment beyond binary classification
- **Polarity ratios** detect mixed or ambiguous sentiment
- **Intensity modifiers** capture degree of sentiment strength

### P14: Semantic Representation
- **Content word density** approximates semantic richness without embeddings
- **Placeholder for future**: Replace TF-IDF with sentence-transformers or GloVe embeddings

### Anaphora Resolution
- **Pronoun patterns** indicate cohesion and narrative structure
- **Noun-pronoun distance** measures referential complexity
- Useful for identifying cohesive vs. fragmented reviews

### RST Summarization
- **Sentence salience** identifies key sentences (first/last/high-sentiment)
- **Discourse markers** reveal rhetorical structure (argumentation, elaboration)
- Can be used to weight sentences in extractive summarization

### Language Identification
- **Function word ratios** vary across languages (e.g., English vs. Romance languages)
- Extension features can detect code-mixing or non-native writing patterns
- Not explicitly implemented but can be tested on multilingual data

## Limitations

1. **No Neural Embeddings**: BERT/GloVe excluded due to computational cost (would require GPU, >4GB models)
2. **Lexicon-based**: Emotion detection uses fixed word lists, missing contextual meanings
3. **English-only**: Discourse markers and pronouns are English-specific
4. **No Deep Anaphora**: Only surface-level pronoun-noun distance, not full coreference resolution

## Future Enhancements

- Add sentence-transformers for lightweight semantic embeddings
- Multilingual pronoun and discourse marker lists
- Neural anaphora resolution using spaCy neuralcoref
- Cross-lingual feature extraction for language identification tasks

## Performance Impact

Extension features add:
- **Computation time**: ~15% overhead (mostly discourse parsing)
- **Feature count**: +28 features (+155% of baseline 18)
- **Expected accuracy gain**: +1-2% for nuanced datasets (e.g., sarcasm, mixed sentiment)

Use `--extensions` flag only when marginal gains justify the cost.
