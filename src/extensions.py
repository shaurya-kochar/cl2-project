#!/usr/bin/env python3

import re
import json
import numpy as np
import pandas as pd
from collections import Counter


EMOTION_LEXICON = {
    'joy': ['happy', 'joy', 'delight', 'pleased', 'wonderful', 'amazing', 'excellent', 'great', 'love', 'fantastic'],
    'anger': ['angry', 'mad', 'furious', 'hate', 'terrible', 'awful', 'worst', 'horrible', 'disgusting', 'annoying'],
    'sadness': ['sad', 'depressed', 'disappointed', 'unhappy', 'unfortunate', 'tragic', 'melancholy', 'gloomy'],
    'fear': ['afraid', 'scared', 'frightened', 'terrified', 'worried', 'anxious', 'nervous', 'creepy', 'scary'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'sudden', 'startled']
}

INTENSITY_WORDS = {
    'very', 'really', 'extremely', 'absolutely', 'completely', 'totally', 
    'quite', 'rather', 'somewhat', 'fairly', 'pretty', 'incredibly'
}

PRONOUNS = {
    'personal': ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours'],
    'demonstrative': ['this', 'that', 'these', 'those'],
    'relative': ['who', 'whom', 'whose', 'which', 'that'],
    'reflexive': ['myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves']
}

DISCOURSE_CONNECTIVES = {
    'contrast': ['however', 'but', 'although', 'though', 'yet', 'nevertheless', 'nonetheless', 'despite'],
    'cause': ['because', 'since', 'therefore', 'thus', 'hence', 'consequently', 'so'],
    'addition': ['moreover', 'furthermore', 'additionally', 'also', 'besides', 'plus'],
    'temporal': ['then', 'meanwhile', 'afterwards', 'subsequently', 'finally', 'eventually'],
    'example': ['for example', 'for instance', 'such as', 'like', 'namely']
}


def extract_emotion_features(tokens_list, lexicon=None):
    if lexicon is None:
        lexicon = load_sentiment_lexicon()
    
    emotion_features = []
    
    for tokens in tokens_list:
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        
        tokens_lower = [t.lower() for t in tokens]
        
        emotion_scores = {}
        for emotion, words in EMOTION_LEXICON.items():
            score = sum(1 for t in tokens_lower if t in words)
            emotion_scores[f'emotion_{emotion}'] = score
        
        polarity_pos = sum(1 for t in tokens_lower if lexicon.get(t, 0) > 0)
        polarity_neg = sum(1 for t in tokens_lower if lexicon.get(t, 0) < 0)
        
        intensity_count = sum(1 for t in tokens_lower if t in INTENSITY_WORDS)
        
        emotion_scores['polarity_positive'] = polarity_pos
        emotion_scores['polarity_negative'] = polarity_neg
        emotion_scores['polarity_ratio'] = polarity_pos / (polarity_neg + 1)
        emotion_scores['intensity_modifiers'] = intensity_count
        
        emotion_features.append(emotion_scores)
    
    return pd.DataFrame(emotion_features)


def extract_anaphora_features(tokens_list, pos_tags_list):
    anaphora_features = []
    
    for tokens, pos_tags in zip(tokens_list, pos_tags_list):
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if isinstance(pos_tags, str):
            pos_tags = json.loads(pos_tags)
        
        tokens_lower = [t.lower() for t in tokens]
        
        pronoun_counts = {}
        for ptype, pwords in PRONOUNS.items():
            count = sum(1 for t in tokens_lower if t in pwords)
            pronoun_counts[f'pronoun_{ptype}'] = count
        
        pronoun_density = sum(pronoun_counts.values()) / len(tokens) if tokens else 0
        
        noun_indices = [i for i, tag in enumerate(pos_tags) if tag.startswith('NN')]
        pronoun_indices = [i for i, t in enumerate(tokens_lower) 
                          if any(t in pwords for pwords in PRONOUNS.values())]
        
        avg_distance = 0
        if pronoun_indices and noun_indices:
            distances = []
            for pidx in pronoun_indices:
                closest_noun = min(noun_indices, key=lambda nidx: abs(nidx - pidx))
                distances.append(abs(closest_noun - pidx))
            avg_distance = np.mean(distances) if distances else 0
        
        pronoun_counts['pronoun_density'] = pronoun_density
        pronoun_counts['pronoun_noun_distance'] = avg_distance
        
        anaphora_features.append(pronoun_counts)
    
    return pd.DataFrame(anaphora_features)


def extract_discourse_salience_features(tokens_list, text_raw_list, lexicon=None):
    if lexicon is None:
        lexicon = load_sentiment_lexicon()
    
    salience_features = []
    
    for tokens, text_raw in zip(tokens_list, text_raw_list):
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text_raw) if s.strip()]
        
        sent_positions = []
        sent_lengths = []
        sent_sentiment_scores = []
        sent_discourse_counts = []
        
        token_idx = 0
        for sent in sentences:
            sent_tokens = sent.lower().split()
            sent_len = len(sent_tokens)
            
            if not sent_tokens:
                continue
            
            position = token_idx / len(tokens) if tokens else 0
            sent_positions.append(position)
            sent_lengths.append(sent_len)
            
            sentiment = sum(lexicon.get(t, 0) for t in sent_tokens)
            sent_sentiment_scores.append(abs(sentiment))
            
            disc_count = sum(1 for t in sent_tokens 
                           if any(t in words for words in DISCOURSE_CONNECTIVES.values()))
            sent_discourse_counts.append(disc_count)
            
            token_idx += sent_len
        
        if sent_positions:
            first_sent_salience = 1.0 - sent_positions[0]
            last_sent_salience = sent_positions[-1]
            max_sentiment_sent_pos = sent_positions[np.argmax(sent_sentiment_scores)]
            
            sent_length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        else:
            first_sent_salience = 0
            last_sent_salience = 0
            max_sentiment_sent_pos = 0
            sent_length_variance = 0
        
        connective_counts = {}
        tokens_lower = [t.lower() for t in tokens]
        for ctype, cwords in DISCOURSE_CONNECTIVES.items():
            count = sum(1 for t in tokens_lower if t in cwords)
            connective_counts[f'discourse_{ctype}'] = count
        
        salience = {
            'first_sent_salience': first_sent_salience,
            'last_sent_salience': last_sent_salience,
            'max_sentiment_position': max_sentiment_sent_pos,
            'sent_length_variance': sent_length_variance,
            'num_sentences': len(sentences),
            **connective_counts
        }
        
        salience_features.append(salience)
    
    return pd.DataFrame(salience_features)


def extract_semantic_density_features(tokens_list, pos_tags_list):
    semantic_features = []
    
    for tokens, pos_tags in zip(tokens_list, pos_tags_list):
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if isinstance(pos_tags, str):
            pos_tags = json.loads(pos_tags)
        
        content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        content_words = [tokens[i] for i, tag in enumerate(pos_tags) if tag in content_tags]
        
        content_density = len(content_words) / len(tokens) if tokens else 0
        content_diversity = len(set(content_words)) / len(content_words) if content_words else 0
        
        function_words = len(tokens) - len(content_words)
        function_ratio = function_words / len(tokens) if tokens else 0
        
        semantic_features.append({
            'content_word_density': content_density,
            'content_word_diversity': content_diversity,
            'function_word_ratio': function_ratio
        })
    
    return pd.DataFrame(semantic_features)


def load_sentiment_lexicon():
    try:
        from nltk.corpus import opinion_lexicon
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())
        lexicon = {}
        for w in pos_words:
            lexicon[w] = 1
        for w in neg_words:
            lexicon[w] = -1
        return lexicon
    except:
        import nltk
        nltk.download('opinion_lexicon', quiet=True)
        from nltk.corpus import opinion_lexicon
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())
        lexicon = {}
        for w in pos_words:
            lexicon[w] = 1
        for w in neg_words:
            lexicon[w] = -1
        return lexicon


def extract_all_extension_features(df, lexicon=None):
    if lexicon is None:
        lexicon = load_sentiment_lexicon()
    
    emotion_df = extract_emotion_features(df['tokens'], lexicon)
    anaphora_df = extract_anaphora_features(df['tokens'], df['pos_tags'])
    salience_df = extract_discourse_salience_features(df['tokens'], df['text_raw'], lexicon)
    semantic_df = extract_semantic_density_features(df['tokens'], df['pos_tags'])
    
    extensions = pd.concat([emotion_df, anaphora_df, salience_df, semantic_df], axis=1)
    return extensions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed_sample.csv")
    parser.add_argument("--output", default="results/extension_features.csv")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Extracting extension features...")
    extensions = extract_all_extension_features(df)
    
    print(f"\nExtension features shape: {extensions.shape}")
    print("\nFeature summary:")
    print(extensions.describe())
    
    extensions.to_csv(args.output, index=False)
    print(f"\nSaved extension features to {args.output}")
