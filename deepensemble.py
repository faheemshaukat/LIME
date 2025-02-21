import pandas as pd
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW
)
from sklearn.metrics import f1_score, accuracy_score, classification_report, multilabel_confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Define genres
genres = ['action', 'adventure', 'comedy', 'crime', 'drama', 'horror', 'romance', 'sci-fi', 'thriller']

# Load data
df_train = pd.read_excel('/kaggle/input/testdata/Split0.xlsx', sheet_name='train')
df_val = pd.read_excel('/kaggle/input/testdata/Split0.xlsx', sheet_name='valid')
df_test = pd.read_excel('/kaggle/input/testdata/new_LMTD9.xlsx', sheet_name='test')

# Combine training and validation data for final training
df_train_val = pd.concat([df_train, df_val], ignore_index=True)

# Tokenize and encode data for each model
def tokenize_data(tokenizer, data, max_length):
    encodings = tokenizer.batch_encode_plus(
        data['plot'].tolist(), 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    labels = torch.tensor(data[genres].values, dtype=torch.float32)
    return input_ids, attention_masks, labels

# Define tokenizers and max_length
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 256

# Tokenize data for each model
input_ids_train_bert, attention_masks_train_bert, labels_train_bert = tokenize_data(bert_tokenizer, df_train_val, max_length)
input_ids_test_bert, attention_masks_test_bert, labels_test_bert = tokenize_data(bert_tokenizer, df_test, max_length)

input_ids_train_distilbert, attention_masks_train_distilbert, labels_train_distilbert = tokenize_data(distilbert_tokenizer, df_train_val, max_length)
input_ids_test_distilbert, attention_masks_test_distilbert, labels_test_distilbert = tokenize_data(distilbert_tokenizer, df_test, max_length)

input_ids_train_roberta, attention_masks_train_roberta, labels_train_roberta = tokenize_data(roberta_tokenizer, df_train_val, max_length)
input_ids_test_roberta, attention_masks_test_roberta, labels_test_roberta = tokenize_data(roberta_tokenizer, df_test, max_length)

# Create DataLoader for each model
batch_size = 32

def create_dataloader(input_ids, attention_masks, labels, batch_size, sampler):
    data = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(data, sampler=sampler(data), batch_size=batch_size)
    return dataloader

train_dataloader_bert = create_dataloader(input_ids_train_bert, attention_masks_train_bert, labels_train_bert, batch_size, RandomSampler)
test_dataloader_bert = create_dataloader(input_ids_test_bert, attention_masks_test_bert, labels_test_bert, batch_size, SequentialSampler)

train_dataloader_distilbert = create_dataloader(input_ids_train_distilbert, attention_masks_train_distilbert, labels_train_distilbert, batch_size, RandomSampler)
test_dataloader_distilbert = create_dataloader(input_ids_test_distilbert, attention_masks_test_distilbert, labels_test_distilbert, batch_size, SequentialSampler)

train_dataloader_roberta = create_dataloader(input_ids_train_roberta, attention_masks_train_roberta, labels_train_roberta, batch_size, RandomSampler)
test_dataloader_roberta = create_dataloader(input_ids_test_roberta, attention_masks_test_roberta, labels_test_roberta, batch_size, SequentialSampler)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(genres))
model_bert.to(device)

model_distilbert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(genres))
model_distilbert.to(device)

model_roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(genres))
model_roberta.to(device)

# Set up optimizer and loss function for each model
def setup_optimizer_and_loss(model):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_func = BCEWithLogitsLoss()
    return optimizer, loss_func

optimizer_bert, loss_func_bert = setup_optimizer_and_loss(model_bert)
optimizer_distilbert, loss_func_distilbert = setup_optimizer_and_loss(model_distilbert)
optimizer_roberta, loss_func_roberta = setup_optimizer_and_loss(model_roberta)

# Training loop for each model
def train_model(model_name, model, optimizer, loss_func, train_dataloader, validation_dataloader, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=masks)
            logits = outputs.logits
            loss = loss_func(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Avg Train Loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, masks, labels = batch
                outputs = model(input_ids, attention_mask=masks)
                logits = outputs.logits
                loss = loss_func(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(validation_dataloader)
        print(f"Avg Validation Loss: {avg_val_loss}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            print(f"Best {model_name} model saved!")

# Train models
train_model("bert", model_bert, optimizer_bert, loss_func_bert, train_dataloader_bert, test_dataloader_bert, epochs=3)
train_model("distilbert", model_distilbert, optimizer_distilbert, loss_func_distilbert, train_dataloader_distilbert, test_dataloader_distilbert, epochs=3)
train_model("roberta", model_roberta, optimizer_roberta, loss_func_roberta, train_dataloader_roberta, test_dataloader_roberta, epochs=3)

# Load best models
model_bert.load_state_dict(torch.load('best_model_bert.pth'))
model_distilbert.load_state_dict(torch.load('best_model_distilbert.pth'))
model_roberta.load_state_dict(torch.load('best_model_roberta.pth'))

# Testing each model
def test_model(model, test_dataloader):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, labels = batch
            outputs = model(input_ids, attention_mask=masks)
            logits = outputs.logits
            preds.extend(logits.sigmoid().cpu().detach().numpy())
            true_labels.extend(labels.cpu().numpy())
    preds_binary = np.array(preds) > 0.5
    true_labels_binary = np.array(true_labels) > 0.5
    return preds_binary, true_labels_binary

preds_test_bert, true_labels_test_bert = test_model(model_bert, test_dataloader_bert)
preds_test_distilbert, true_labels_test_distilbert = test_model(model_distilbert, test_dataloader_distilbert)
preds_test_roberta, true_labels_test_roberta = test_model(model_roberta, test_dataloader_roberta)

# Evaluate each model
print("BERT Classification Report:")
print(classification_report(true_labels_test_bert, preds_test_bert, target_names=genres))

print("DistilBERT Classification Report:")
print(classification_report(true_labels_test_distilbert, preds_test_distilbert, target_names=genres))

print("RoBERTa Classification Report:")
print(classification_report(true_labels_test_roberta, preds_test_roberta, target_names=genres))

# Hard Voting Ensemble
def hard_voting_ensemble_multilabel(models, dataloader):
    all_predictions = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, masks, _ = batch
                outputs = model(input_ids, attention_mask=masks)
                logits = outputs.logits
                preds.extend(logits.sigmoid().cpu().detach().numpy())
        all_predictions.append(np.array(preds) > 0.5)  # Convert to binary predictions

    # Apply hard voting: for each sample and each class, take the majority vote
    ensemble_preds = np.sum(all_predictions, axis=0) > (len(models) // 2)
    return ensemble_preds

# Perform hard voting ensemble
ensemble_models = [model_bert, model_distilbert, model_roberta]
ensemble_preds_test_hard = hard_voting_ensemble_multilabel(ensemble_models, test_dataloader_bert)

# Evaluate hard voting ensemble
print("Hard Voting Ensemble Classification Report:")
print(classification_report(true_labels_test_bert, ensemble_preds_test_hard, target_names=genres))

# Soft Voting Ensemble
def soft_voting_ensemble_multilabel(models, dataloader):
    all_predictions = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, masks, _ = batch
                outputs = model(input_ids, attention_mask=masks)
                logits = outputs.logits
                preds.extend(logits.sigmoid().cpu().detach().numpy())
        all_predictions.append(np.array(preds))  # Keep probabilities for soft voting

    # Apply soft voting: average probabilities across models
    ensemble_preds = np.mean(all_predictions, axis=0) > 0.5
    return ensemble_preds

# Perform soft voting ensemble
ensemble_preds_test_soft = soft_voting_ensemble_multilabel(ensemble_models, test_dataloader_bert)

# Evaluate soft voting ensemble
print("Soft Voting Ensemble Classification Report:")
print(classification_report(true_labels_test_bert, ensemble_preds_test_soft, target_names=genres))

# Save soft voting ensemble models and tokenizers
def save_ensemble_models(models, tokenizers, model_names):
    for model, tokenizer, name in zip(models, tokenizers, model_names):
        # Save model
        model.save_pretrained(f'ensemble_model_{name}')
        # Save tokenizer
        tokenizer.save_pretrained(f'ensemble_model_{name}')
    print("Soft voting ensemble models and tokenizers saved!")

# Define model names
model_names = ['bert', 'distilbert', 'roberta']

# Save ensemble models
save_ensemble_models([model_bert, model_distilbert, model_roberta],
                     [bert_tokenizer, distilbert_tokenizer, roberta_tokenizer],
                     model_names)

# Plot ROC and Precision-Recall Curves
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

# Plot curves for hard voting ensemble
plot_roc_curve(true_labels_test_bert, ensemble_preds_test_hard, genres, 'Hard Voting Ensemble')
plot_precision_recall_curve(true_labels_test_bert, ensemble_preds_test_hard, genres, 'Hard Voting Ensemble')

# Plot curves for soft voting ensemble
plot_roc_curve(true_labels_test_bert, ensemble_preds_test_soft, genres, 'Soft Voting Ensemble')
plot_precision_recall_curve(true_labels_test_bert, ensemble_preds_test_soft, genres, 'Soft Voting Ensemble')