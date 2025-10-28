# Knowledge Based Case Retrieval

##  English Text Filtering Script

### **Overview**

This script cleans a text dataset by **removing all non-English lines** from files in a given folder. It uses `langdetect` for language detection and saves only English content into a new output folder.

---

### **Usage**

1. Install dependencies:

   ```bash
   pip install langdetect tqdm
   ```
2. Set folder paths in the script:

   ```python
   input_folder = "data/task1_train_files_2025"
   output_folder = "data/processed_train_langonly"
   ```
   Then for test
   ```python
   input_folder = "data/task1_test_files_2025"
   output_folder = "data/processed_test_langonly"
   ```
4. Run:

  Run file preprocess.ipynb

---

## Legal Case Embedding Pipeline

A modular pipeline to process, clean, and embed legal case documents for further query candidate pair feature making. This project automates the preparation of legal case text data and generates dense embeddings at multiple granularities using transformer models.

## 📁 Folder Structure

```
project_root/
│
├── data/
│   ├── processed_train_langonly/
│   ├── processed_test_langonly/
│   ├── task1_train_labels_2025.json
│   ├── task1_test_no_labels_2025.json
│
├── embeddings_output/
│   ├── embeddings_sentences_en.npy
│   ├── embeddings_paragraphs_formatted.npy
│   ├── embeddings_propositions_en.npy
│   └── embeddings_final.pkl
│
├── file_code.py
├── model_code.py
├── pairs_code.py
└── new_main.ipynb
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* Transformers
* Pandas, NumPy, tqdm

Install dependencies:

```bash
pip install torch transformers pandas numpy tqdm
```

---

**Embedding Generation**
   Uses `sentence-transformers/all-MiniLM-L6-v2` to compute embeddings for each text component.

## 📦 Outputs

| File                                  | Description                            |
| ------------------------------------- | -------------------------------------- |
| `embeddings_sentences_en.npy`         | Sentence-level embeddings              |
| `embeddings_paragraphs_formatted.npy` | Paragraph-level embeddings             |
| `embeddings_propositions_en.npy`      | Proposition-level embeddings           |
| `embeddings_final.pkl`                | Combined DataFrame with all embeddings |

---

Perfect — ab tumhara **BM25-based candidate generation pipeline** complete ho gaya hai ✅
Let’s write a clean, professional **README.md** (without unnecessary code), focused on the *BM25 retrieval and pair generation* stage of your workflow.

---

## BM25 Candidate Retrieval for Legal Case Matching

This module retrieves top candidate legal cases for each query paragraph using **BM25**, then builds structured query–candidate pairs for downstream model training. After preprocessing and embedding generation, this stage applies **BM25 ranking** to identify the most textually similar candidate cases for each paragraph (or proposition) of a query case. The output is a compact set of high-relevance candidates (e.g., top-50 per paragraph) — essential for efficient fine-tuning and evaluation of retrieval or re-ranking models. Performs round-robin sampling to retain up to **100 unique candidates per query**


## 📁 Input Requirements

| File                                  | Description                       |
| ------------------------------------- | --------------------------------- |
| `files.pkl` or `files` DataFrame      | Processed and cleaned legal cases |
| `data/task1_train_labels_2025.json`   | Query–target training label map   |
| `data/task1_test_no_labels_2025.json` | Test queries (unlabeled)          |
| `rank_bm25` library                   | For BM25-based text ranking       |

---

## ⚙️ Dependencies

Install required libraries:

```bash
pip install rank_bm25 nltk spacy pandas tqdm
python -m spacy download en_core_web_sm
```

## 🧮 Output Files

| Output                        | Description                                                                             |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `bm25_results_para_train.csv` | Top-50 BM25 candidates per query paragraph (train set)                                  |
| `bm25_results_para_test.csv`  | Top-50 BM25 candidates per query paragraph (test set)                                   |
| `pairs_para_100_unique.pkl`   | Final dataset with up to 100 unique BM25 candidates per query and binary `match` labels |


## Feature Engineering for Legal Case Pairs

This module generates **feature vectors** for query–candidate case pairs using both **semantic embeddings** and **textual similarity metrics**.
These features are later used to train supervised models (e.g., classifiers or rankers) to predict whether two cases are **relevant (`match=1`)** or **irrelevant (`match=0`)**.

---

### 🧩 Inputs

| File                                                  | Description                                  |
| ----------------------------------------------------- | -------------------------------------------- |
| `pairs_para_100_unique.pkl`                           | Query–candidate pairs (BM25-generated)       |
| `embeddings_new_2_without_prop_afterpropparasent.pkl` | Processed file-level embeddings and metadata |

---

Each step in `pairs_code.py` adds or computes a specific set of features:


### 📦 Output

| File                           | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| `features_para_100_unique.pkl` | Final feature-enriched DataFrame (ready for ML model input) |


## Legal Case Retrieval — Model Training & Evaluation

This stage trains and evaluates the final retrieval model that predicts whether a **query–candidate** pair is relevant (`match=1`) or not.

---

## 📁 Inputs

* `pickle/features_para_100_unique.pkl` — feature-enriched pairs
* `data/task1_train_labels_2025.json`, `data/task1_test_labels_2025.json` — ground truth labels
* `model_code.py` — model functions
* `output/results_2.txt` — final predictions

## 📦 Outputs

* `experiments/train_prediction_bm25.txt`
* `experiments/test_prediction_bm25.txt`
* `output/results_2.txt` (final model predictions)

## Evaluation Summary

This section presents the **retrieval performance** of the trained model under different inference settings.
Each configuration varies by **inference type (1 or 2)**, **case granularity (full vs. paragraph-wise)**, and **ranking cutoff (Top-100 / Top-300)**.

---

## 📁 Dataset Statistics

| Split     | Queries | Candidates |
| :-------- | :-----: | :--------: |
| **Train** |  1,678  |    5,452   |
| **Test**  |   400   |    1,759   |

---

## 🧠 Evaluation Settings

| Setting             | Description                                                   |
| :------------------ | :------------------------------------------------------------ |
| **Inference 1 / 2** | Two variants of inference strategy for candidate ranking      |
| **Full Cases**      | Model uses embeddings of entire cases                         |
| **Paragraph-wise**  | Model aggregates similarity scores across paragraphs          |
| **Top-K**           | Evaluation performed for top-100 or top-300 ranked candidates |

---

## 📈 Results Overview

| Inference |             Granularity            | Top-K | Precision |   Recall  |  F1-score |
| :-------: | :--------------------------------: | :---: | :-------: | :-------: | :-------: |
|     1     |              Full case             |  300  |   0.305   |   0.336   |   0.320   |
|     2     |              Full case             |  300  |   0.324   |   0.317   |   0.320   |
|     1     |           Paragraph-wise           |  100  |   0.312   |   0.351   |   0.330   |
|     2     |           Paragraph-wise           |  100  |   0.325   |   0.345   |   0.335   |
|     1     | Paragraph-wise (top-100 corrected) |  100  |   0.324   |   0.366   |   0.344   |
|     2     | Paragraph-wise (top-100 corrected) |  100  | **0.347** | **0.349** | **0.348** |
|     1     |              Full case             |  100  |   0.303   |   0.313   |   0.308   |
|     2     |              Full case             |  100  |   0.327   |   0.295   |   0.310   |

---

## 🏁 Key Insights

* **Inference-2 consistently outperforms** Inference-1 across settings.
* **Paragraph-wise evaluation** yields better F1-scores than full-case inference.
* **Best overall performance:**
  🔹 *Inference-2 (Paragraph-wise, Top-100)* → **F1 = 0.348**

---
