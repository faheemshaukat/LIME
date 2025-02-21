# README: Movie Genre Classification Using Ensemble Learning

This repository contains the code for a multi-label movie genre classification system using ensemble learning techniques. The system combines traditional machine learning algorithms (Naive Bayes, SVM, and Logistic Regression) with soft and hard voting to classify movie plots into multiple genres. The model is trained on the Trailer12K dataset and tested on both the Trailer12K and LMTD9 datasets.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Model Architecture](#model-architecture)
5. [Implementation Steps](#implementation-steps)
6. [Results](#results)
7. [License](#license)

---

## Introduction

The goal of this project is to classify movie plots into multiple genres using an ensemble of traditional machine learning algorithms. The system employs a combination of Naive Bayes, Support Vector Machines (SVM), and Logistic Regression, with soft and hard voting mechanisms to improve classification accuracy. The model is trained on the Trailer12K dataset and evaluated on both the Trailer12K and LMTD9 datasets.

---

## Dataset

The following datasets are used in this project:
1. **Trailer12K**: A dataset containing movie plots and their corresponding genres. The dataset is split into three parts (Split 0, Split 1, and Split 2) for cross-validation.
2. **LMTD9**: An external dataset used for testing the model's generalization ability.

### Dataset Structure
- **Training Data**: `Split0.xlsx`, `Split1.xlsx`, `Split2.xlsx` (sheets: `train`, `valid`)
- **Testing Data**: `new_LMTD9.xlsx` (sheet: `test`)

---

## Preprocessing Steps

Several preprocessing steps were applied to clean and standardize the text data:
1. **Punctuation Cleaning**: Removed punctuation that does not contribute to the meaning of the plot (e.g., commas, periods).
2. **Special Character Removal**: Removed irrelevant symbols like `@` or `#`.
3. **Contraction Expansion**: Expanded contractions (e.g., "didn't" becomes "did not") to maintain consistency.
4. **Stop Word Reduction**: Minimized commonly used words (e.g., "the," "is") without completely removing them.
5. **Redundancy Removal**: Filtered out repetitive phrases or terms.
6. **Spacing Consistency**: Corrected spacing inconsistencies for uniformity.

### Example of Preprocessing
| **Preprocessing Step**              | **Movie Plot**                                                                                   |
|-------------------------------------|--------------------------------------------------------------------------------------------------|
| **Original Plot**                   | John, a brilliant scientist, embarks on a dangerous journey to stop an evil organization...      |
| **Punctuation Cleaning**            | John a brilliant scientist embarks on a dangerous journey to stop an evil organization...        |
| **Contraction Expansion**           | John a brilliant scientist embarks on a dangerous journey to stop an evil organization...        |
| **Stop Word Reduction**             | John brilliant scientist embarks dangerous journey stop evil organization...                     |
| **Redundancy Removal & Consistency**| John brilliant scientist embarks dangerous journey stop evil organization...                     |

---

## Model Architecture

The model uses an ensemble of three traditional machine learning algorithms:
1. **Naive Bayes (NB)**: A probabilistic classifier based on Bayes' theorem.
2. **Support Vector Machine (SVM)**: A classifier that finds the optimal hyperplane for separating classes.
3. **Logistic Regression (LR)**: A linear model for binary and multi-class classification.

### Ensemble Techniques
- **Soft Voting**: Combines the predicted probabilities of each classifier.
- **Hard Voting**: Combines the predicted class labels of each classifier.

### Multi-Output Classifier
The `MultiOutputClassifier` is used to handle multi-label classification, where each movie plot can belong to multiple genres.

---

## Implementation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/faheemshaukat/LIME
cd LIME
```

### Step 2: Install Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib tqdm openpyxl
```

### Step 3: Prepare the Dataset
1. Place the dataset files (`Split0.xlsx`, `Split1.xlsx`, `Split2.xlsx`, and `new_LMTD9.xlsx`) in the `data` folder.
2. Update the file paths in the code if necessary.

### Step 4: Run the Code
Execute the Python script to train and evaluate the model:
```bash
python movie_genre_classification.py
```

### Step 5: View Results
- The classification reports, confusion matrices, and AUC metrics will be printed in the console.
- ROC and Precision-Recall curves will be saved as PNG files in the working directory.
- Predicted results for the test dataset will be saved as Excel files.

---

## Results

### Evaluation Metrics
- **Classification Report**: Precision, recall, and F1-score for each genre.
- **Confusion Matrix**: Multi-label confusion matrix for each genre.
- **AUC Scores**: Micro, macro, weighted, and sample average AUC scores.

### Graphs
1. **ROC Curve**: Shows the trade-off between true positive rate (TPR) and false positive rate (FPR).
2. **Precision-Recall Curve**: Shows the trade-off between precision and recall.
3. **Combined Curves**: Combines ROC and Precision-Recall curves for all genres.

### Example Output
```
Classification Report for Soft Voting Classifier:
              precision    recall  f1-score   support
       action       0.85      0.78      0.81       500
    adventure       0.82      0.75      0.78       450
       comedy       0.88      0.82      0.85       600
        crime       0.79      0.71      0.75       400
        drama       0.83      0.77      0.80       550
       horror       0.81      0.74      0.77       350
      romance       0.84      0.79      0.81       500
       sci-fi       0.80      0.73      0.76       450
      thriller       0.82      0.76      0.79       500
```

---

## License

This project is licensed under the GIT 2025 License.

---

## Acknowledgments
- The Trailer12K and LMTD9 datasets were used for training and testing.
- The ensemble learning approach was implemented using scikit-learn.
```