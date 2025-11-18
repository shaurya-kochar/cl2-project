# IMDb Dataset Instructions

This project uses the **IMDB Dataset of 50K Movie Reviews** for sentiment classification.

## Dataset Details

- **Size**: 50,000 reviews (25,000 positive, 25,000 negative)
- **File**: `IMDB_Dataset.csv`
- **Columns**: `review` (text), `sentiment` (positive/negative)
- **File size**: ~63 MB (not included in git repository)

## Download Options

### Option 1: Kaggle (Recommended)

1. Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Download `IMDB Dataset.csv`
3. Rename to `IMDB_Dataset.csv` and place in this directory (`data/`)

### Option 2: HuggingFace Datasets (Programmatic)

```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("imdb")

# Convert to pandas and save
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
full_df = pd.concat([train_df, test_df], ignore_index=True)

# Map labels: 0 -> negative, 1 -> positive
full_df['sentiment'] = full_df['label'].map({0: 'negative', 1: 'positive'})
full_df = full_df.rename(columns={'text': 'review'})
full_df = full_df[['review', 'sentiment']]

# Save
full_df.to_csv('data/IMDB_Dataset.csv', index=False)
print(f"Saved {len(full_df)} reviews to data/IMDB_Dataset.csv")
```

### Option 3: Direct Download (Alternative Mirror)

```bash
# Using wget (Linux/Mac)
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

# Then use the conversion script (not provided, but similar to Option 2)
```

## Verification

After downloading, verify the file:

```bash
ls -lh data/IMDB_Dataset.csv
# Should show ~63 MB

head -n 3 data/IMDB_Dataset.csv
# Should show: review,sentiment header and 2 data rows
```

## Processed Files

After running preprocessing, you will see:
- `processed.csv` - Full preprocessed dataset (~353 MB, not in git)
- `meta.json` - Dataset statistics and metadata

**Note**: Large CSV files are excluded from git via `.gitignore`. Only the raw `IMDB_Dataset.csv` needs to be manually downloaded.
