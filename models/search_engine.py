"""
CS410 FA25 Final Project
Group 18
"""
import re
import math
import pickle
import numpy as np
import pandas as pd
import torch
from collections import Counter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)


class SearchEngine:
    """BM25F retrieval + semantic reranking"""

    def __init__(self, alpha=0.7, w_plot=0.6, w_kw=0.3, w_meta=0.1, stage1_k=200):
        self.alpha = alpha
        self.w_plot = w_plot
        self.w_kw = w_kw
        self.w_meta = w_meta
        self.stage1_k = stage1_k

        # Will be initialized on load
        self.df = None
        self.bm25_plot = None
        self.bm25_keywords = None
        self.bm25_meta = None
        self.embeddings = None
        self.semantic_model = None
        self.idf_scores = None
        self.corpus_full = None

        # Setup NLP
        self._setup_nlp()

    def _setup_nlp(self):
        STOPWORDS = set(stopwords.words('english'))
        KEEP_WORDS = {'not', 'no', 'never', 'without', 'none'}
        self.stopwords = STOPWORDS - KEEP_WORDS
        self.nlp = spacy.load('en_core_web_sm', disable=["parser"])

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[\\,<>./?@#$%^&*_~!()\-=\[\]{};:\'\"\|`0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.ent_type_:
                tokens.append(token.text)
            elif token.lemma_ not in self.stopwords and len(token.text) > 2:
                tokens.append(token.lemma_)
        return " ".join(tokens)

    def load_data(self, csv_path, embeddings_path=None, idf_path=None, preprocessed=False):
        print("Loading dataset...")
        self.df = pd.read_csv(csv_path)

        # Check if all cleaned columns exist
        has_all_cleaned = all(col in self.df.columns for col in ['plot_clean', 'keywords_clean', 'meta_clean', 'full_text'])

        if preprocessed and has_all_cleaned:
            print("Using fully preprocessed data (skipping text cleaning)...")
        else:
            print("Preprocessing text fields...")
            self.df['plot_clean'] = self.df['synopsis'].apply(lambda x: self.clean_text(str(x)))
            self.df['keywords_clean'] = self.df.apply(
                lambda r: self.clean_text(f"{r.get('Key-Bert', '')} {r.get('Yake', '')} {r.get('Sentence_transformers', '')}"),
                axis=1
            )
            self.df['meta_clean'] = self.df.apply(
                lambda r: self.clean_text(f"{r.get('genre', '')} {r.get('year', '')}"),
                axis=1
            )
            self.df['full_text'] = self.df['plot_clean'] + ' ' + self.df['keywords_clean'] + ' ' + self.df['meta_clean']

        # Build BM25 indices
        print("Building BM25 indices...")
        corpus_plot = [doc.split() for doc in self.df['plot_clean']]
        corpus_keywords = [doc.split() for doc in self.df['keywords_clean']]
        corpus_meta = [doc.split() for doc in self.df['meta_clean']]
        self.corpus_full = [doc.split() for doc in self.df['full_text']]

        self.bm25_plot = BM25Okapi(corpus_plot, k1=1.2, b=0.75)
        self.bm25_keywords = BM25Okapi(corpus_keywords, k1=1.2, b=0.75)
        self.bm25_meta = BM25Okapi(corpus_meta, k1=1.2, b=0.75)

        # Load or compute embeddings
        print("Loading semantic model...")
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        if embeddings_path and torch.cuda.is_available():
            try:
                self.embeddings = torch.load(embeddings_path)
                print(f"Loaded embeddings from {embeddings_path}")
            except:
                self._compute_embeddings()
        else:
            if embeddings_path:
                try:
                    self.embeddings = torch.load(embeddings_path, map_location='cpu')
                    print(f"Loaded embeddings from {embeddings_path}")
                except:
                    self._compute_embeddings()
            else:
                self._compute_embeddings()

        # Load or compute IDF scores
        if idf_path:
            try:
                with open(idf_path, 'rb') as f:
                    self.idf_scores = pickle.load(f)
                print(f"Loaded IDF scores from {idf_path}")
            except:
                self._compute_idf()
        else:
            self._compute_idf()

        print(f"Search engine ready with {len(self.df)} movies")

    def _compute_embeddings(self):
        print("Computing embeddings (this may take a few minutes)...")
        self.embeddings = self.semantic_model.encode(
            self.df['full_text'].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def _compute_idf(self):
        print("Computing IDF scores...")
        N = len(self.corpus_full)
        df_counts = Counter()
        for doc in self.corpus_full:
            df_counts.update(set(doc))
        self.idf_scores = {term: math.log((N + 1) / (count + 1)) for term, count in df_counts.items()}

    def _bm25f_score(self, query):
        """Compute weighted BM25 scores across fields."""
        q_tokens = self.clean_text(query).split()
        scores_plot = self.bm25_plot.get_scores(q_tokens)
        scores_kw = self.bm25_keywords.get_scores(q_tokens)
        scores_meta = self.bm25_meta.get_scores(q_tokens)
        return self.w_plot * scores_plot + self.w_kw * scores_kw + self.w_meta * scores_meta

    def _normalize_scores(self, scores):
        """Min-max normalize scores."""
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _expand_query_prf(self, query, top_docs_idx, n_terms=10):
        """Expand query using pseudo-relevance feedback."""
        term_scores = Counter()
        for idx in top_docs_idx:
            doc_terms = self.corpus_full[idx]
            tf = Counter(doc_terms)
            for term, count in tf.items():
                if term in self.idf_scores:
                    term_scores[term] += count * self.idf_scores[term]

        original_terms = set(self.clean_text(query).split())
        expansion_terms = [t for t, _ in term_scores.most_common(n_terms + len(original_terms))
                         if t not in original_terms][:n_terms]

        expanded = self.clean_text(query) + ' ' + ' '.join(expansion_terms)
        return expanded

    def search(self, query, top_k=10, use_prf=True, prf_docs=10):
        """
        Two-stage search:
        1. BM25F retrieves candidates
        2. Semantic reranking on candidates
        """
        q_clean = self.clean_text(query)

        # PRF expansion
        if use_prf:
            initial_scores = self._bm25f_score(query)
            initial_idx = np.argsort(initial_scores)[::-1][:prf_docs].copy()
            q_expanded = self._expand_query_prf(query, initial_idx)
        else:
            q_expanded = q_clean

        # Stage 1: BM25F retrieval
        bm25_scores = self._bm25f_score(q_expanded if use_prf else query)
        candidate_idx = np.argsort(bm25_scores)[::-1][:self.stage1_k].copy()

        # Stage 2: Semantic reranking
        candidate_embeddings = self.embeddings[candidate_idx]
        q_emb = self.semantic_model.encode(q_clean, convert_to_tensor=True)
        sem_scores = util.cos_sim(q_emb, candidate_embeddings)[0].cpu().numpy()

        # Combine scores
        bm25_norm = self._normalize_scores(bm25_scores[candidate_idx])
        sem_norm = self._normalize_scores(sem_scores)
        hybrid_scores = self.alpha * sem_norm + (1 - self.alpha) * bm25_norm

        # Get top-k results
        top_in_candidates = np.argsort(hybrid_scores)[::-1][:top_k].copy()
        final_idx = candidate_idx[top_in_candidates]
        final_scores = hybrid_scores[top_in_candidates]

        # Format results
        results = []
        for idx, score in zip(final_idx, final_scores):
            row = self.df.iloc[idx]
            results.append({
                'title': row['movie_title'],
                'year': str(row.get('year', 'N/A')),
                'genre': str(row.get('genre', '')).replace(', Back to top', ''),
                'synopsis': str(row.get('synopsis', ''))[:500],
                'score': float(score)
            })

        return results

    def save_model(self, embeddings_path, idf_path, config_path):
        """Save model components."""
        torch.save(self.embeddings, embeddings_path)
        with open(idf_path, 'wb') as f:
            pickle.dump(self.idf_scores, f)
        config = {
            'alpha': self.alpha,
            'w_plot': self.w_plot,
            'w_kw': self.w_kw,
            'w_meta': self.w_meta,
            'stage1_k': self.stage1_k
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        print(f"Model saved to {embeddings_path}, {idf_path}, {config_path}")


def create_search_engine(data_dir='data', models_dir='models'):
    """Factory function to create and load search engine."""
    import os

    engine = SearchEngine(alpha=0.7, w_plot=0.6, w_kw=0.3, w_meta=0.1)

    # Use preprocessed dataset for faster loading
    csv_path = os.path.join(data_dir, 'imdb_movie_keyword_preprocessed.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, 'imdb_movie_keyword.csv')
        preprocessed = False
    else:
        preprocessed = True

    embeddings_path = os.path.join(models_dir, 'embeddings.pt')
    idf_path = os.path.join(models_dir, 'idf_scores.pkl')

    engine.load_data(csv_path, embeddings_path, idf_path, preprocessed=preprocessed)

    return engine
