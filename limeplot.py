import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

# Define genres
genres = ['action', 'adventure', 'comedy', 'crime', 'drama', 'fantasy', 'horror', 'romance', 'sci-fi', 'thriller']

# Load the saved soft ensemble models
def load_ensemble_models(model_names):
    models = []
    tokenizers = []
    for name in model_names:
        if name == 'bert':
            model = BertForSequenceClassification.from_pretrained(f'ensemble_model_{name}')
            tokenizer = BertTokenizer.from_pretrained(f'ensemble_model_{name}')
        elif name == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(f'ensemble_model_{name}')
            tokenizer = DistilBertTokenizer.from_pretrained(f'ensemble_model_{name}')
        elif name == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained(f'ensemble_model_{name}')
            tokenizer = RobertaTokenizer.from_pretrained(f'ensemble_model_{name}')
        models.append(model)
        tokenizers.append(tokenizer)
    return models, tokenizers

# Load ensemble models and tokenizers
model_names = ['bert', 'distilbert', 'roberta']
ensemble_models, ensemble_tokenizers = load_ensemble_models(model_names)

# Move models to CPU for LIME explanations
for model in ensemble_models:
    model.to('cpu')

# Class for ensemble predictions (for LIME)
class EnsemblePredictor:
    def __init__(self, models, tokenizers, max_length):
        self.models = models
        self.tokenizers = tokenizers
        self.max_length = max_length

    def __call__(self, texts):
        all_preds = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            encodings = tokenizer.batch_encode_plus(
                texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids']  # Keep inputs on CPU
            attention_mask = encodings['attention_mask']  # Keep attention masks on CPU

            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits).numpy()  # Convert to numpy directly (since we're on CPU)
            all_preds.append(preds)

        # Average predictions across models for soft voting
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds

# Initialize the LIME explainer
explainer = LimeTextExplainer(class_names=genres)

# Updated genre colors with HEX values
genre_colors = {
    'action': '#1f7712',      # RGB (31, 119, 18)
    'adventure': '#aec702',   # RGB (174, 199, 2)
    'comedy': '#ff7f01',      # RGB (255, 127, 1)
    'crime': '#ffbb01',       # RGB (255, 187, 1)
    'drama': '#2ca004',       # RGB (44, 160, 4)
    'fantasy': '#98df01',     # RGB (152, 223, 1)
    'horror': '#d62704',      # RGB (214, 39, 4)
    'romance': '#FFC0CB',     # Pink (Hex: #FFC0CB)
    'sci-fi': '#946701',      # RGB (148, 103, 1)
    'thriller': '#c5b002',    # RGB (197, 176, 2)
}

# Color for negative scores: Light black (Dark grey)
negative_score_color = '#505050'  # Lighter shade of black

# Function to explain prediction for a specified genre index with LIME and save as PNG
def explain_genre(ensemble_predictor, plot, genre_index):
    # Get the predictions for the plot
    preds = ensemble_predictor([plot])[0]

    # Explain the prediction for the specified genre using LIME
    explanation = explainer.explain_instance(
        plot, 
        ensemble_predictor, 
        num_samples=500,  # Reduce perturbation samples for efficiency
        num_features=10,  # Number of features to display
        labels=[genre_index]  # Only show explanation for specified genre
    )

    # Get the genre name
    genre_name = genres[genre_index].lower()  # Use lower to match genre_colors keys
    
    print(f"\nExplanation for genre: {genre_name.capitalize()}")

    # Show LIME's default explanation plot
    explanation.show_in_notebook(text=plot)

    # Save the LIME explanation to an HTML file
    html_file_path = f'lime_explanation_{genre_name}.html'  # Customize the filename if desired
    explanation.save_to_file(html_file_path)
    print(f"LIME explanation saved to {html_file_path}")

    # Extract feature importance and create a bar plot using custom colors
    exp_map = explanation.as_list(label=genre_index)
    words, scores = zip(*exp_map)  # Unpack the words and their importance scores

    # Assign colors: each genre gets its own color for positive scores, light black for negative scores
    colors = [genre_colors[genre_name] if score > 0 else negative_score_color for score in scores]

    # Plot the LIME explanation using matplotlib with custom colors
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(scores)), scores, color=colors)  # Use the colors based on score
    plt.yticks(range(len(words)), words, fontsize=12)
    plt.xlabel("Importance", fontsize=14)
    plt.title(f"LIME Explanation for {genre_name.capitalize()}", fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance on top

    # Save bar chart as PNG with 600 DPI
    plt.savefig(f'lime_explanation_{genre_name}.png', dpi=600, bbox_inches='tight')  # Save with high resolution
    plt.show()

# Tokenize a sample from the test set for LIME explanation
df_test = pd.read_excel('/kaggle/input/testdata/Split2.xlsx', sheet_name='test')  # Load your test data
plot_to_explain = df_test['plot'].iloc[1]  # Change this index to analyze different plots

# Create the ensemble predictor
max_length = 256  # Set your max_length as required
ensemble_predictor = EnsemblePredictor(ensemble_models, ensemble_tokenizers, max_length)

# Get predictions for the plot
preds = ensemble_predictor([plot_to_explain])[0]

# Sort genres by prediction probability and get the top 2 genres
top_genre_indices = np.argsort(preds)[-2:][::-1]  # Get indices of the top 2 genres

# Display top genres and their probabilities
for i, genre_idx in enumerate(top_genre_indices):
    print(f"Top {i + 1} predicted genre: {genres[genre_idx]} with probability: {preds[genre_idx]:.4f}")

# Explain the top 1 predicted genre using LIME
print("\nExplaining the top 1 predicted genre:")
explain_genre(ensemble_predictor, plot_to_explain, top_genre_indices[0])

# Explain the second predicted genre using LIME
print("\nExplaining the second predicted genre:")
explain_genre(ensemble_predictor, plot_to_explain, top_genre_indices[1])