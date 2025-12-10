# UIUCinema

Plot-based movie search engine using hybrid retrieval (BM25 + semantic reranking).

**CS410 FA25 Final Project - Group 18**

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/praneeth0045/imdb-top-10000-movie-plots-keywords) and place `imdb_movie_keyword.csv` in `data/`.

### 3. Train model
Open `Train.ipynb` and run all cells. Set `LOAD_FROM_FILES = False` in the first cell to train from scratch.

This will:
- Preprocess the dataset
- Build BM25 indices
- Compute semantic embeddings
- Tune hyperparameters with Optuna
- Save model files to `models/`

### 4. Run webapp
```bash
python app.py
```
Open http://localhost:5000

## Project Structure

```
UIUCinema/
├── app.py              # Flask webapp
├── Train.ipynb         # Training and evaluation
├── data/
│   └── imdb_movie_keyword.csv
├── models/
│   ├── search_engine.py
│   ├── embeddings.pt
│   ├── idf_scores.pkl
│   └── model_config.pkl
└── templates/
```

## Team

- Theo Xiong ([@TheoXiong7](https://github.com/TheoXiong7))
- Yichong Liu ([@YiChong-Liu](https://github.com/YiChong-Liu))
- Mei Han ([@fishbrook](https://github.com/fishbrook))
