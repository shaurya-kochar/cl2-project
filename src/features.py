#!/usr/bin/env python3

import json
import pickle
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


DISCOURSE_MARKERS = {'however', 'but', 'therefore', 'thus', 'moreover', 'furthermore', 
                     'although', 'though', 'nevertheless', 'nonetheless', 'meanwhile'}

NEGATIONS = {"no", "not", "never", "n't", "nothing", "nowhere", "neither", "nor", "none"}

AFINN_LEXICON = None


def load_afinn_lexicon():
    global AFINN_LEXICON
    if AFINN_LEXICON is not None:
        return AFINN_LEXICON
    
    try:
        from nltk.corpus import opinion_lexicon
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())
        lexicon = {}
        for w in pos_words:
            lexicon[w] = 1
        for w in neg_words:
            lexicon[w] = -1
        AFINN_LEXICON = lexicon
    except:
        nltk.download('opinion_lexicon', quiet=True)
        from nltk.corpus import opinion_lexicon
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())
        lexicon = {}
        for w in pos_words:
            lexicon[w] = 1
        for w in neg_words:
            lexicon[w] = -1
        AFINN_LEXICON = lexicon
    
    return AFINN_LEXICON


class FeatureExtractor:
    def __init__(self, tfidf_max_features=20000, tfidf_ngram=(1,1)):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram = tfidf_ngram
        self.tfidf_vectorizer = None
        self.lexicon = None
        
    def fit_tfidf(self, texts):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram,
            min_df=2,
            max_df=0.95,
            lowercase=True
        )
        self.tfidf_vectorizer.fit(texts)
        return self
    
    def transform_tfidf(self, texts):
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted")
        return self.tfidf_vectorizer.transform(texts)
    
    def extract_pos_features(self, pos_tags_list):
        pos_features = []
        for pos_tags in pos_tags_list:
            if isinstance(pos_tags, str):
                pos_tags = json.loads(pos_tags)
            
            total = len(pos_tags) if pos_tags else 1
            counter = Counter(pos_tags)
            
            features = {
                'pos_noun_pct': counter.get('NN', 0) / total + counter.get('NNS', 0) / total + counter.get('NNP', 0) / total,
                'pos_verb_pct': counter.get('VB', 0) / total + counter.get('VBD', 0) / total + counter.get('VBG', 0) / total + counter.get('VBN', 0) / total + counter.get('VBP', 0) / total + counter.get('VBZ', 0) / total,
                'pos_adj_pct': counter.get('JJ', 0) / total + counter.get('JJR', 0) / total + counter.get('JJS', 0) / total,
                'pos_adv_pct': counter.get('RB', 0) / total + counter.get('RBR', 0) / total + counter.get('RBS', 0) / total,
                'pos_pron_pct': counter.get('PRP', 0) / total + counter.get('PRP$', 0) / total
            }
            pos_features.append(features)
        
        return pd.DataFrame(pos_features)
    
    def extract_negation_features(self, tokens_list, pos_tags_list):
        neg_features = []
        
        for tokens, pos_tags in zip(tokens_list, pos_tags_list):
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            if isinstance(pos_tags, str):
                pos_tags = json.loads(pos_tags)
            
            neg_count = sum(1 for t in tokens if t.lower() in NEGATIONS)
            
            neg_adj_count = 0
            for i, tok in enumerate(tokens):
                if tok.lower() in NEGATIONS:
                    for j in range(i+1, min(i+4, len(tokens))):
                        if pos_tags[j] in ['JJ', 'JJR', 'JJS']:
                            neg_adj_count += 1
                            break
            
            neg_features.append({
                'negation_count': neg_count,
                'neg_adj_window': neg_adj_count
            })
        
        return pd.DataFrame(neg_features)
    
    def extract_sentiment_lexicon_features(self, tokens_list):
        if self.lexicon is None:
            self.lexicon = load_afinn_lexicon()
        
        sent_features = []
        for tokens in tokens_list:
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            
            scores = [self.lexicon.get(t.lower(), 0) for t in tokens]
            scores = [s for s in scores if s != 0]
            
            features = {
                'lex_score_sum': sum(scores) if scores else 0,
                'lex_score_mean': np.mean(scores) if scores else 0,
                'lex_score_max': max(scores) if scores else 0,
                'lex_score_min': min(scores) if scores else 0
            }
            sent_features.append(features)
        
        return pd.DataFrame(sent_features)
    
    def extract_stylistic_features(self, df):
        features = []
        
        for _, row in df.iterrows():
            text_raw = row.get('text_raw', '')
            tokens = row.get('tokens', [])
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            
            sentences = [s.strip() for s in re.split(r'[.!?]+', text_raw) if s.strip()]
            
            exclaim_count = text_raw.count('!')
            question_count = text_raw.count('?')
            
            features.append({
                'avg_word_len': row.get('avg_word_len', 0),
                'avg_sent_len': len(tokens) / len(sentences) if sentences else 0,
                'punct_intensity': exclaim_count + question_count,
                'ttr': row.get('ttr', 0)
            })
        
        return pd.DataFrame(features)
    
    def extract_discourse_features(self, tokens_list):
        disc_features = []
        
        for tokens in tokens_list:
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            
            tokens_lower = [t.lower() for t in tokens]
            
            first_sent_idx = 0
            for i, t in enumerate(tokens):
                if t in '.!?':
                    first_sent_idx = i
                    break
            
            first_sent_tokens = tokens_lower[:first_sent_idx] if first_sent_idx > 0 else tokens_lower[:min(20, len(tokens_lower))]
            
            if self.lexicon is None:
                self.lexicon = load_afinn_lexicon()
            
            first_sent_score = sum(self.lexicon.get(t, 0) for t in first_sent_tokens)
            
            disc_marker_count = sum(1 for t in tokens_lower if t in DISCOURSE_MARKERS)
            
            disc_features.append({
                'first_sent_sentiment': first_sent_score,
                'discourse_markers': disc_marker_count
            })
        
        return pd.DataFrame(disc_features)
    
    def save_vectorizer(self, path):
        if self.tfidf_vectorizer is None:
            raise ValueError("No vectorizer to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
    
    def load_vectorizer(self, path):
        with open(path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        return self


def extract_all_features(df, extractor, include_tfidf=True):
    feature_dfs = []
    
    if include_tfidf:
        tfidf_matrix = extractor.transform_tfidf(df['text_clean'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        feature_dfs.append(tfidf_df)
    
    pos_df = extractor.extract_pos_features(df['pos_tags'])
    feature_dfs.append(pos_df)
    
    neg_df = extractor.extract_negation_features(df['tokens'], df['pos_tags'])
    feature_dfs.append(neg_df)
    
    lex_df = extractor.extract_sentiment_lexicon_features(df['tokens'])
    feature_dfs.append(lex_df)
    
    style_df = extractor.extract_stylistic_features(df)
    feature_dfs.append(style_df)
    
    disc_df = extractor.extract_discourse_features(df['tokens'])
    feature_dfs.append(disc_df)
    
    result = pd.concat(feature_dfs, axis=1)
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed_sample.csv")
    parser.add_argument("--output", default="results/feature_summary.csv")
    parser.add_argument("--vectorizer-out", default="models/tfidf_vectorizer.pkl")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram", type=str, default="1,1")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    ngram = tuple(map(int, args.ngram.split(',')))
    extractor = FeatureExtractor(tfidf_max_features=args.max_features, tfidf_ngram=ngram)
    
    print("Fitting TF-IDF vectorizer...")
    extractor.fit_tfidf(df['text_clean'])
    
    print("Extracting all features...")
    features = extract_all_features(df, extractor, include_tfidf=False)
    
    print("\nFeature Summary:")
    summary = features.describe().T
    summary.to_csv(args.output)
    print(f"Saved feature summary to {args.output}")
    
    extractor.save_vectorizer(args.vectorizer_out)
    print(f"Saved TF-IDF vectorizer to {args.vectorizer_out}")
