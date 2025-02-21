import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc
from tqdm import tqdm
import re
import string

# Start timer
start_time = time.time()

# Load the Excel file into a pandas DataFrame
df_train = pd.read_excel('/kaggle/input/testdata/Split0.xlsx', sheet_name='train')
df_val = pd.read_excel('/kaggle/input/testdata/Split0.xlsx', sheet_name='valid')
df_test = pd.read_excel('/kaggle/input/testdata/new_LMTD9.xlsx', sheet_name='test')

# Combine training and validation data for final training
df_train_val = pd.concat([df_train, df_val], ignore_index=True)

# Split data into features (movie plots) and labels (genres)
genres = ['action', 'adventure', 'comedy', 'crime', 'drama', 'horror', 'romance', 'sci-fi', 'thriller'] #Here is 9 Genre for LMTD9 incase Trailer12k use 'Fantasy' also
X_train_val = df_train_val['plot']
y_train_val = df_train_val[genres]

X_test = df_test['plot']
y_test = df_test[genres]

# Data Preprocessing
"""
Several pre-processing steps were performed to clean and standardize the text:
1. Punctuation Cleaning: Removed punctuation that does not contribute to the meaning of the plot (e.g., commas, periods).
2. Special Character Removal: Removed irrelevant symbols like @ or #.
3. Contraction Expansion: Expanded contractions (e.g., "didn't" becomes "did not") to maintain consistency.
4. Stop Word Reduction: Minimized commonly used words (e.g., "the," "is") without completely removing them.
5. Redundancy Removal: Filtered out repetitive phrases or terms.
6. Spacing Consistency: Corrected spacing inconsistencies for uniformity.
"""

def preprocess_text(text):
    # Step 1: Punctuation Cleaning
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # Step 2: Special Character Removal
    text = re.sub(r"[@#\$%&\*\(\)\{\}\[\]\<\>]", "", text)
    
    # Step 3: Contraction Expansion
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "it's": "it is",
        "didn't": "did not",
        "don't": "do not",
        "isn't": "is not",
        "wasn't": "was not",
        "hasn't": "has not",
        "haven't": "have not",
        "aren't": "are not",
        "weren't": "were not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "doesn't": "does not",
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would"
    }
    for contraction, expansion in contractions.items():
        text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
    
    # Step 4: Stop Word Reduction
    stop_words = set(["the", "is", "and", "a", "an", "in", "it", "to", "of", "for", "on", "with", "as", "at", "by", "this", "that", "these", "those"])
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    
    # Step 5: Redundancy Removal and Spacing Consistency
    text = " ".join(text.split())  # Remove extra spaces
    
    return text

# Apply preprocessing to the text data
print("Preprocessing text data...")
X_train_val = X_train_val.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Transform movie plots into TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_val_tfidf = vectorizer.fit_transform(X_train_val)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize individual classifiers
nb = MultinomialNB()
svm = SVC(kernel='linear', probability=True)
lr = LogisticRegression(max_iter=1000)

# Create a voting classifier for soft voting
voting_clf_soft = VotingClassifier(estimators=[
    ('nb', nb),
    ('svm', svm),
    ('lr', lr)
], voting='soft')

# Create a voting classifier for hard voting
voting_clf_hard = VotingClassifier(estimators=[
    ('nb', nb),
    ('svm', svm),
    ('lr', lr)
], voting='hard')

# MultiOutputClassifier for multi-label classification
multi_voting_clf_soft = MultiOutputClassifier(voting_clf_soft, n_jobs=-1)
multi_voting_clf_hard = MultiOutputClassifier(voting_clf_hard, n_jobs=-1)

# Fit the classifiers
print("Fitting the soft voting model...")
multi_voting_clf_soft.fit(X_train_val_tfidf, y_train_val)

print("Fitting the hard voting model...")
multi_voting_clf_hard.fit(X_train_val_tfidf, y_train_val)

# Make predictions with both soft and hard voting classifiers
print("Making predictions...")
preds_soft = multi_voting_clf_soft.predict(X_test_tfidf)
preds_hard = multi_voting_clf_hard.predict(X_test_tfidf)

# Get predicted probabilities for soft voting (required for ROC and PR curves)
preds_proba_soft = multi_voting_clf_soft.predict_proba(X_test_tfidf)
y_pred_proba_soft = np.array([proba[:, 1] for proba in preds_proba_soft]).T

# Generate classification report
print("Generating classification report for soft voting...")
report_soft = classification_report(y_test, preds_soft, target_names=genres, output_dict=True)
report_str_soft = classification_report(y_test, preds_soft, target_names=genres)
print("Classification Report for Soft Voting Classifier:")
print(report_str_soft)

print("Generating classification report for hard voting...")
report_hard = classification_report(y_test, preds_hard, target_names=genres, output_dict=True)
report_str_hard = classification_report(y_test, preds_hard, target_names=genres)
print("Classification Report for Hard Voting Classifier:")
print(report_str_hard)

# Generate multilabel confusion matrix
print("Generating confusion matrix for soft voting...")
cm_soft = multilabel_confusion_matrix(y_test, preds_soft)
print("Confusion Matrix for Soft Voting Classifier:")
print(cm_soft)

print("Generating confusion matrix for hard voting...")
cm_hard = multilabel_confusion_matrix(y_test, preds_hard)
print("Confusion Matrix for Hard Voting Classifier:")
print(cm_hard)

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred, genres, classifier_name):
    plt.figure(figsize=(12, 8))
    for i, genre in enumerate(genres):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, lw=2, label=f'{genre} (area = {auc(recall, precision):.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {classifier_name}')
    plt.legend(loc='best')
    plt.savefig(f'{classifier_name}_precision_recall_curve.png', dpi=600)
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred, genres, classifier_name):
    plt.figure(figsize=(15, 10))
    for i, genre in enumerate(genres):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{genre} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {classifier_name}')
    plt.legend(loc='best')
    plt.savefig(f'{classifier_name}_roc_curve.png', dpi=600)
    plt.show()

# Function to plot combined ROC and PR curves
def plot_combined_curves(y_true, y_pred, classifier_name):
    fpr_combined, tpr_combined, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc_combined = auc(fpr_combined, tpr_combined)

    precision_combined, recall_combined, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    pr_auc_combined = auc(recall_combined, precision_combined)

    plt.figure(figsize=(14, 7))

    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr_combined, tpr_combined, label=f'ROC curve (area = {roc_auc_combined:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Genres Combined)')
    plt.legend()

    # Plot PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_combined, precision_combined, label=f'PR curve (AP = {pr_auc_combined:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (All Genres Combined)')
    plt.legend()

    plt.savefig(f'{classifier_name}_combined_curves.png', dpi=600)
    plt.show()

# Plot precision-recall and ROC curves for soft voting
plot_precision_recall_curve(y_test.values, y_pred_proba_soft, genres, 'Soft Voting Classifier')
plot_roc_curve(y_test.values, y_pred_proba_soft, genres, 'Soft Voting Classifier')
plot_combined_curves(y_test.values, y_pred_proba_soft, 'Soft Voting Classifier')

# End timer and print total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")