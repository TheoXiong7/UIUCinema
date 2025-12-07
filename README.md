# UIUCinema

Plot-based movie search engine using two-stage hybrid retrieval (BM25F + semantic reranking).

**CS410 FA25 Final Project - Group 18**

## Project Structure

```
UIUCinema/
├── app.py                 # Flask webapp
├── preprocessing.py       # preprocess dataset
├── Train.ipynb            # train and evaluate model
├── data/
│   ├── imdb_movie_keyword.csv
│   └── imdb_movie_keyword_preprocessed.csv
├── models/
│   ├── search_engine.py   # hybrid search engine
│   ├── embeddings.pt      # pre-computed sentence embeddings
│   └── idf_scores.pkl     # IDF scores for PRF
└── templates/             # HTML templates
```

## Setup

**Download Dataset**
```
https://www.kaggle.com/datasets/praneeth0045/imdb-top-10000-movie-plots-keywords?resource=download
```

**Install Requirements**
```bash
pip install -r requirements.txt
```

## Usage

**Preprocess data:**
```bash
python preprocessing.py
```

**Train/evaluate model:**
```bash
jupyter notebook Train.ipynb
```

**Run web app:**
```bash
python app.py
```
Then open http://localhost:5000

## Model

1. **BM25F Retrieval** - Field-weighted scoring (plot, keywords, metadata) with pseudo-relevance feedback
2. **Semantic Reranking** - Top-200 candidates reranked using all-MiniLM-L6-v2 embeddings
3. **Hybrid Score** - `alpha * semantic + (1-alpha) * BM25` with alpha=0.7
