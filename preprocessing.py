"""Preprocessing script for UIUCinema dataset."""
import re
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
KEEP_WORDS = {'not', 'no', 'never', 'without', 'none'}
STOPWORDS = STOPWORDS - KEEP_WORDS

nlp = spacy.load('en_core_web_sm', disable=["parser"])


def clean_text(text):
    """Clean, normalize, and lemmatize text."""
    text = str(text).lower()
    text = re.sub(r'[\\,<>./?@#$%^&*_~!()\-=\[\]{};:\'\"\|`0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.ent_type_:
            tokens.append(token.text)
        elif token.lemma_ not in STOPWORDS and len(token.text) > 2:
            tokens.append(token.lemma_)
    return " ".join(tokens)


def preprocess_dataset(input_path='data/imdb_movie_keyword.csv',
                       output_path='data/imdb_movie_keyword_preprocessed.csv'):
    """Preprocess dataset and save with cleaned fields."""
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} movies")

    print("Cleaning plot text...")
    df['plot_clean'] = df['synopsis'].apply(lambda x: clean_text(str(x)))

    print("Cleaning keywords...")
    df['keywords_clean'] = df.apply(
        lambda r: clean_text(f"{r.get('Key-Bert', '')} {r.get('Yake', '')} {r.get('Sentence_transformers', '')}"),
        axis=1
    )

    print("Cleaning metadata...")
    df['meta_clean'] = df.apply(
        lambda r: clean_text(f"{r.get('genre', '')} {r.get('year', '')}"),
        axis=1
    )

    print("Creating full text...")
    df['full_text'] = df['plot_clean'] + ' ' + df['keywords_clean'] + ' ' + df['meta_clean']

    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Saved {len(df)} records")

    return df


if __name__ == '__main__':
    preprocess_dataset()
