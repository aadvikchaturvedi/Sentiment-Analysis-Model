# Emotion Classification from Text (TF-IDF + XGBoost)

A six-class emotion classifier for short English text. It labels each sentence as **sadness, joy, love, anger, fear or surprise**. The full pipeline lives in one Colab notebook, `code.ipynb`: data download, cleaning, TF-IDF features, an XGBoost multi-class model and evaluation.

Test accuracy is **87.44%** on 83,225 held-out sentences.

## Data

The data is the Kaggle dataset [`bhavikjikadara/emotions-dataset`](https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset), downloaded with `kagglehub`. It contains 416,809 rows of `text` and `label`. After removing 686 duplicate rows, 416,123 remain.

| Label | Emotion | Count |
|---|---|---|
| 0 | sadness | 121,187 |
| 1 | joy | 141,067 |
| 2 | love | 34,554 |
| 3 | anger | 57,317 |
| 4 | fear | 47,712 |
| 5 | surprise | 14,972 |

The classes are imbalanced: joy is about 9.4 times larger than surprise.

## Pipeline

```
emotions.csv ─► drop duplicates ─► stratified 80/20 split (random_state=42)
                                         │
              preprocess: strip URLs and @mentions, keep letters only,
                          lowercase, remove NLTK stopwords, Porter stemming
                                         │
              TfidfVectorizer(max_features=5000), fitted on train only
                                         │
              XGBClassifier(objective='multi:softmax', num_class=6)
                                         │
              accuracy + confusion matrix on the test split
```

- Train set: 332,898 sentences. Test set: 83,225. Both are 5,000-dimensional sparse TF-IDF matrices.
- The vectorizer is fitted on the training split only, so no test vocabulary or IDF statistics reach the model.
- XGBoost uses its default tree settings. Nothing is tuned.

## Results

Test accuracy is **0.8744**. Per-class recall below is computed from the notebook's confusion matrix.

| Emotion | Test samples | Correct | Recall |
|---|---|---|---|
| sadness | 24,198 | 21,781 | 0.90 |
| joy | 28,156 | 25,197 | 0.89 |
| love | 6,899 | 5,245 | 0.76 |
| anger | 11,447 | 9,954 | 0.87 |
| fear | 9,533 | 7,892 | 0.83 |
| surprise | 2,992 | 2,703 | 0.90 |

The two largest error groups make sense for these labels:
- Love and joy are often confused (1,306 love→joy, 1,424 joy→love).
- Fear is often predicted as surprise (741).

Bag-of-stems features can't see the context that separates these pairs.

## How to run

1. Open `code.ipynb` in Google Colab (there is a badge at the top of the notebook).
2. Run all cells. `kagglehub` downloads the dataset into its cache. If Colab asks for Kaggle credentials, provide your own.

To run locally:

```bash
pip install kagglehub pandas nltk scikit-learn xgboost matplotlib seaborn
jupyter notebook code.ipynb
```

The notebook loads the CSV from a hard-coded path, `/root/.cache/kagglehub/datasets/bhavikjikadara/emotions-dataset/versions/1`. When running outside Colab, change that path to the one printed by the `kagglehub.dataset_download` cell.

## Limitations / next steps

- Accuracy is the only aggregate metric reported. Given the class imbalance, macro-F1 would be a more useful summary.
- The model, vectorizer and preprocessing are not saved. There is no `predict()` helper, CLI or API, so the only way to classify new text is to re-run the notebook.
- Stemming and stopword removal throw away negation ("not happy" becomes "happi") and word order.
- No class weighting or hyperparameter search.
- Natural next steps:
  - Persist the pipeline (`joblib`) and add a small inference script.
  - Compare against a linear baseline (logistic regression on the same TF-IDF features).
  - Compare against a fine-tuned transformer such as DistilBERT.

## License

MIT
