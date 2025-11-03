
# 🧠 KBC: Knowledge-Based Case Retrieval

This repository contains code and experiments for a **Knowledge-Based Case (KBC) Retrieval system**.
The project explores different approaches — including ensemble models and text-to-text models like T5 — to retrieve and evaluate knowledge-based cases from structured and unstructured inputs.

---

## 📁 Project Structure

| File / Folder                | Description                                                                                                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`main.ipynb`**             | The main implementation notebook. Handles the core KBC retrieval pipeline without relying on T5. This is the primary approach for case retrieval and evaluation.     |
| **`file_code.py`**           | Contains the **ensemble model** logic — combines predictions or outputs from different models to improve retrieval accuracy.                                         |
| **`output_inference.ipynb`** | Used for **testing and evaluation** — applies the model outputs on test labels to generate and verify the final retrieval results.                                   |
| **`utils.ipynb`**            | Includes utility functions to **merge multiple output files** and to integrate additional proposition or related field information (used in T5-related experiments). |
| **`model_code.py`**          | Defines the core **model architecture and methods** used for retrieval (e.g., vector representations, similarity computation, etc.).                                 |
| **`pairs_code.py`**          | Handles **data pairing or preparation** logic — creates input-output pairs used for model training or testing.                                                       |
| **`t5train_code.py`**        | Contains the training logic for the **T5 experiment**, where the T5 model is fine-tuned for proposition and relation extraction.                                     |
| **`files/`**                 | Contains input or intermediate data files used during training or testing.                                                                                           |
| **`models/trained_t5/`**     | Stores trained T5 model checkpoints (from T5-based experiments).                                                                                                     |
| **`output/`**                | Directory for generated outputs, predictions, and evaluation results.                                                                                                |
| **`requirements.txt`**       | Lists all Python dependencies needed to run the code.                                                                                                                |
| **`.gitignore`**             | Specifies files and directories (like `.csv` outputs or large models) to be ignored by Git.                                                                          |
| **`README.md`**              | Project documentation (this file).                                                                                                                                   |

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/kritiarora2003/information_retrieval_kbc.git
cd information_retrieval_kbc
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### **Main KBC Implementation**

Run the main retrieval pipeline:

```bash
jupyter notebook main.ipynb
```

### **Ensemble Model**

Combine outputs from different retrieval models:

```bash
python file_code.py
```

### **Output Inference**

Evaluate results on test data:

```bash
jupyter notebook output_inference.ipynb
```

### **T5 Experiment (Optional)**

Train and test T5-based proposition model:

```bash
python t5train_code.py
```

### **Utility Scripts**

Merge outputs and process results:

```bash
jupyter notebook utils.ipynb
```

---

## 🧩 Overview

The **KBC Retrieval System** aims to:

* Retrieve similar or related cases based on input queries.
* Experiment with multiple architectures, including ensemble and T5-based methods.
* Evaluate retrieval quality using structured test labels.
* Support post-processing and merging of model outputs.

---
