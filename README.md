# fake-essay-detection
This repository contains a machine learning model designed to detect whether an essay was written by a student or by a large language model (LLM).

## Problem Statement

With the growing sophistication of LLMs, it's becoming increasingly difficult to distinguish between human-written and AI-generated text. This project aims to develop a reliable tool to help educators and content creators identify potentially machine-generated essays.
## Methodology

1.Preprocessing:
  * Text Cleaning: Removal of noise, normalization (e.g., lowercasing).
  * Tokenization: Using Byte-Pair Encoding (BPE) for efficient handling of out-of-vocabulary words.

2.Feature Engineering:
  * TF-IDF Vectorization: Converting text into numerical representations, emphasizing the importance of words within a document.
3.Modeling:
  * Ensemble Classifier: Combining multiple models (Multinomial Naive Bayes, SGDClassifier, LightGBM, CatBoost) with weighted voting for robust predictions.
4.Evaluation:
 * Metric: ROC-AUC score to assess performance on imbalanced datasets.

## Usage

**Dependencies:**

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* LightGBM
* CatBoost
* Transformers
* Datasets
* tqdm

**Installation:**
  `pip install -r requirements.txt`

## Results

The model achieved an accuracy of 0.95% and a ROC-AUC score of 0.986 on the test set.

## Contributing

Contributions to improve this project are welcome. Please follow these guidelines:

* Open an issue: Discuss new features or potential improvements.
* Fork the repository and create a pull request for code changes.
