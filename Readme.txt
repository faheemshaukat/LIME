# README: Movie Genre Classification Using Ensemble Learning

This repository contains the code for a multi-label movie genre classification system using **traditional machine learning (ML) algorithms**, **transformer-based deep learning models (BERT, DistilBERT, RoBERTa)**, and **LIME (Local Interpretable Model-agnostic Explanations)** for explainability. The system is trained on the Trailer12K dataset and tested on both the Trailer12K and LMTD9 datasets. The transformer-based soft voting ensemble model is saved for future use, such as explainability with LIME.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Model Architecture](#model-architecture)
5. [Implementation Steps](#implementation-steps)
6. [Results](#results)
7. [LIME Implementation](#lime-implementation)
8. [License](#license)

---

## Introduction

The goal of this project is to classify movie plots into multiple genres using **three approaches**:
1. **Traditional Machine Learning (ML) Ensemble**: Combines Naive Bayes, Support Vector Machines (SVM), and Logistic Regression with soft and hard voting mechanisms.
2. **Transformer-Based Deep Learning Ensemble**: Combines BERT, DistilBERT, and RoBERTa with soft and hard voting mechanisms.
3. **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local and global explanations for the transformer-based ensemble model predictions.

The system is trained on the Trailer12K dataset and evaluated on both the Trailer12K and LMTD9 datasets. The transformer-based soft voting ensemble model is saved for future use, such as explainability with LIME.

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

### Traditional Machine Learning (ML) Ensemble
The traditional ML ensemble uses a combination of three algorithms:
1. **Naive Bayes (NB)**: A probabilistic classifier based on Bayes' theorem.
2. **Support Vector Machine (SVM)**: A classifier that finds the optimal hyperplane for separating classes.
3. **Logistic Regression (LR)**: A linear model for binary and multi-class classification.

#### Ensemble Techniques
- **Soft Voting**: Combines the predicted probabilities of each classifier.
- **Hard Voting**: Combines the predicted class labels of each classifier.

### Transformer-Based Deep Learning Ensemble
The transformer-based ensemble uses a combination of three state-of-the-art models:
1. **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based model pre-trained on large text corpora.
2. **DistilBERT**: A distilled version of BERT, smaller and faster while retaining most of BERT's performance.
3. **RoBERTa (Robustly Optimized BERT Pretraining Approach)**: An optimized version of BERT with improved training techniques.

#### Ensemble Techniques
- **Soft Voting**: Combines the predicted probabilities of each transformer model.
- **Hard Voting**: Combines the predicted class labels of each transformer model.

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
pip install pandas numpy scikit-learn matplotlib tqdm openpyxl torch transformers lime
```

### Step 3: Prepare the Dataset
1. Place the dataset files (`Split0.xlsx`, `Split1.xlsx`, `Split2.xlsx`, and `new_LMTD9.xlsx`) in the `data` folder.
2. Update the file paths in the code if necessary.

### Step 4: Run the Code
Execute the Python scripts to train and evaluate the models:
1. **Traditional ML Ensemble**: Run `tradensemble.py`.
2. **Transformer-Based Deep Learning Ensemble**: Run `deepensemble.py`.
3. **LIME Implementation**: Run `limeplot.py`.

```bash
python tradensemble.py
python deepensemble.py
python limeplot.py
```

### Step 5: View Results
- The classification reports, confusion matrices, and AUC metrics will be printed in the console.
- ROC and Precision-Recall curves will be saved as PNG files in the working directory.
- Predicted results for the test dataset will be saved as Excel files.
- The transformer-based soft voting ensemble model will be saved for future use (e.g., LIME).

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

## LIME Implementation

The `limeplot.py` script provides local and global explanations for the transformer-based ensemble model predictions using LIME. It includes the following functionality:

1. **Loading Ensemble Models**:
   - Loads pre-trained BERT, DistilBERT, and RoBERTa models and their tokenizers.

2. **Ensemble Predictor**:
   - Combines predictions from the transformer models using soft voting.

3. **LIME Explainer**:
   - Initializes a `LimeTextExplainer` to generate explanations for individual predictions.

4. **Explanation Generation**:
   - For a given movie plot, the script generates LIME explanations for the top 2 predicted genres.
   - Displays the explanations in the notebook and saves them as HTML and PNG files.

5. **Custom Visualization**:
   - Creates a bar plot of feature importance using custom colors for positive and negative contributions.

### Example Usage
```python
# Explain the top 1 predicted genre
explain_genre(ensemble_predictor, plot_to_explain, top_genre_indices[0])

# Explain the second predicted genre
explain_genre(ensemble_predictor, plot_to_explain, top_genre_indices[1])
```

### Output
- **HTML File**: Contains the LIME explanation for the specified genre.
- **PNG File**: A high-resolution bar plot showing the most important words for the predicted genre.

---

## License

This project is licensed under the GIT/ZENODO 2025 License.Cite: https://doi.org/10.5281/zenodo.14906135

---

## Acknowledgments
- The Trailer12K and LMTD9 datasets were used for training and testing.
- The traditional ML ensemble approach was implemented using scikit-learn.
- The transformer-based ensemble learning approach was implemented using the `transformers` library by Hugging Face.
- LIME implementation was done using the `lime` library.

---

This README provides a comprehensive overview of the project, including the traditional ML ensemble, transformer-based deep learning ensemble, and LIME implementation for explainability. Let me know if you need further adjustments!