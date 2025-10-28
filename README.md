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
