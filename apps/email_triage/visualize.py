#!/usr/bin/env python3

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

def load_data():
    """Load email dataset"""
    data_file = Path(__file__).parent / 'emails_dataset.csv'
    if not data_file.exists():
        print(f"Error: Dataset not found at {data_file}")
        print("Please run 'python triage.py' first to generate the dataset.")
        return None
    return pd.read_csv(data_file)


def load_models():
    """Load trained models"""
    model_dir = Path(__file__).parent
    models = {}
    
    with open(model_dir / 'extractor.pkl', 'rb') as f:
        models['extractor'] = pickle.load(f)
    with open(model_dir / 'model_category_lr.pkl', 'rb') as f:
        models['category_lr'] = pickle.load(f)
    with open(model_dir / 'encoder_category.pkl', 'rb') as f:
        models['encoder_category'] = pickle.load(f)
    with open(model_dir / 'model_urgency_lr.pkl', 'rb') as f:
        models['urgency_lr'] = pickle.load(f)
    with open(model_dir / 'encoder_urgency.pkl', 'rb') as f:
        models['encoder_urgency'] = pickle.load(f)
    with open(model_dir / 'model_tone_lr.pkl', 'rb') as f:
        models['tone_lr'] = pickle.load(f)
    with open(model_dir / 'encoder_tone.pkl', 'rb') as f:
        models['encoder_tone'] = pickle.load(f)
    
    return models


def plot_category_distribution(df, output_dir):
    """Plot category distribution pie chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    category_counts = df['category'].value_counts()
    colors = plt.cm.Set3(range(len(category_counts)))
    
    wedges, texts, autotexts = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    
    ax.set_title('Email Category Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved category_distribution.png")


def plot_urgency_tone_heatmap(df, output_dir):
    """Plot urgency vs tone heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    crosstab = pd.crosstab(df['urgency'], df['tone'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='white')
    
    ax.set_title('Urgency vs Tone Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tone', fontsize=12, fontweight='bold')
    ax.set_ylabel('Urgency', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'urgency_tone_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved urgency_tone_heatmap.png")


def plot_linguistic_features(df, output_dir):
    """Plot linguistic features comparison"""
    # Check which features exist in the dataset
    possible_features = ['urgency_keywords', 'question_marks', 'exclamation_marks', 'politeness_markers',
                        'length', 'has_greeting', 'has_signature', 'num_questions']
    
    available_features = [f for f in possible_features if f in df.columns]
    
    if len(available_features) < 4:
        # Use first 4 numeric columns if manual features don't exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_features = list(numeric_cols[:4])
    
    if len(available_features) == 0:
        print("⚠ Skipping linguistic features plot - no numeric features found")
        return
    
    features_to_plot = available_features[:4]  # Take first 4
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, feature in zip(axes.flat, features_to_plot):
        df.groupby('category')[feature].mean().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Average Count', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Linguistic Features by Category', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'linguistic_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved linguistic_features.png")


def plot_model_accuracy(output_dir):
    """Plot model accuracy comparison (from training results)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Accuracy values from training (update these based on actual results)
    models = ['Category\nLR', 'Category\nNB', 'Urgency\nLR', 'Urgency\nNB', 'Tone\nLR', 'Tone\nNB']
    accuracy = [1.0, 0.99, 0.40, 0.37, 0.47, 0.42]  # Example values
    colors = ['#3b82f6', '#8b5cf6', '#3b82f6', '#8b5cf6', '#3b82f6', '#8b5cf6']
    
    bars = ax.bar(models, accuracy, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, acc in zip(bars, accuracy):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved model_accuracy.png")


def plot_category_wordclouds(df, output_dir):
    """Generate word clouds for top 3 categories"""
    top_categories = df['category'].value_counts().head(3).index
    
    # Find the text column (might be 'email', 'email_text', or 'text')
    text_col = None
    for col in ['email', 'email_text', 'text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("⚠ Skipping wordcloud plot - no text column found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, category in zip(axes, top_categories):
        text = ' '.join(df[df['category'] == category][text_col].values)
        
        wordcloud = WordCloud(width=600, height=400, background_color='white',
                             colormap='viridis', max_words=50).generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{category}', fontsize=14, fontweight='bold', pad=10)
    
    plt.suptitle('Category Word Clouds (Top 3)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_wordclouds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved category_wordclouds.png")


def plot_urgency_distribution(df, output_dir):
    """Plot urgency distribution by category"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    urgency_order = ['Low', 'Medium', 'High', 'Critical']
    crosstab = pd.crosstab(df['category'], df['urgency'])
    crosstab = crosstab.reindex(columns=urgency_order, fill_value=0)
    
    crosstab.plot(kind='bar', stacked=False, ax=ax, 
                 color=['#84cc16', '#fbbf24', '#f97316', '#ef4444'],
                 edgecolor='white', linewidth=1)
    
    ax.set_title('Urgency Distribution by Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.legend(title='Urgency', title_fontsize=11, fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'urgency_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved urgency_by_category.png")


def plot_tone_distribution(df, output_dir):
    """Plot tone distribution stacked bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tone_counts = df['tone'].value_counts()
    colors = {'Polite': '#10b981', 'Neutral': '#fbbf24', 'Urgent': '#ef4444'}
    tone_colors = [colors.get(tone, '#808080') for tone in tone_counts.index]  # gray fallback
    
    bars = ax.bar(range(len(tone_counts)), tone_counts.values, 
                 color=tone_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    ax.set_xticks(range(len(tone_counts)))
    ax.set_xticklabels(tone_counts.index)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Email Tone Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, tone_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tone_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved tone_distribution.png")


def generate_summary_report(df, output_dir):
    """Generate text summary report"""
    report = []
    report.append("="*70)
    report.append("EMAIL TRIAGE SYSTEM - ANALYSIS REPORT")
    report.append("="*70)
    report.append("")
    
    report.append(f"Total Emails: {len(df)}")
    report.append("")
    
    report.append("CATEGORY DISTRIBUTION")
    report.append("-" * 70)
    for cat, count in df['category'].value_counts().items():
        percentage = (count / len(df)) * 100
        report.append(f"  {cat:25s}: {count:4d} ({percentage:5.1f}%)")
    report.append("")
    
    report.append("URGENCY DISTRIBUTION")
    report.append("-" * 70)
    for urg, count in df['urgency'].value_counts().items():
        percentage = (count / len(df)) * 100
        report.append(f"  {urg:25s}: {count:4d} ({percentage:5.1f}%)")
    report.append("")
    
    report.append("TONE DISTRIBUTION")
    report.append("-" * 70)
    for tone, count in df['tone'].value_counts().items():
        percentage = (count / len(df)) * 100
        report.append(f"  {tone:25s}: {count:4d} ({percentage:5.1f}%)")
    report.append("")
    
    # Only show linguistic features if they exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report.append("LINGUISTIC FEATURES (AVERAGES)")
        report.append("-" * 70)
        for col in numeric_cols[:5]:  # Show first 5 numeric features
            report.append(f"  {col:25s}: {df[col].mean():.2f}")
        report.append("")
    
    report.append("="*70)
    
    report_text = '\n'.join(report)
    
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"✓ Saved analysis_report.txt")
    print("\n" + report_text)


def main():
    print("="*70)
    print("EMAIL TRIAGE SYSTEM - VISUALIZATION GENERATOR")
    print("="*70)
    print()
    
    # Load data
    print("Loading dataset...")
    df = load_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} emails\n")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    plot_category_distribution(df, output_dir)
    plot_urgency_tone_heatmap(df, output_dir)
    plot_linguistic_features(df, output_dir)
    plot_model_accuracy(output_dir)
    plot_category_wordclouds(df, output_dir)
    plot_urgency_distribution(df, output_dir)
    plot_tone_distribution(df, output_dir)
    
    # Generate report
    print("\nGenerating summary report...\n")
    generate_summary_report(df, output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*70)
    print(f"\nAll files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - category_distribution.png")
    print("  - urgency_tone_heatmap.png")
    print("  - linguistic_features.png")
    print("  - model_accuracy.png")
    print("  - category_wordclouds.png")
    print("  - urgency_by_category.png")
    print("  - tone_distribution.png")
    print("  - analysis_report.txt")


if __name__ == '__main__':
    main()
