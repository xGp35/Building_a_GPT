import re
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


# --- Data Preprocessing ---
# Redesigned to work for both training and inference.
# Training mode: DataPreprocessing(file_path) — builds vocabulary from data
# Inference mode: DataPreprocessing(file_path, token_map=existing_token_map) — reuses training vocabulary
# Single sentence: data.tokenize_sentence("some sentence") — uses stored token_map

class DataPreprocessing:
    def __init__(self, file_path: str, max_len: int = 10, token_map=None):
        sentences, labels = self._load_file(file_path)
        sentences = self._clean_sentences(sentences)

        if token_map is None:
            # Training mode: build vocabulary from the data
            self.token_map = self._build_vocab(sentences)
        else:
            # Inference mode: reuse the training vocabulary
            self.token_map = token_map

        self.X = self._tokenize_batch(sentences, max_len)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len

    def _load_file(self, file_path):
        """Load a tab-separated file with header. Returns (sentences_tuple, labels_list)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            data = [line.strip().split('\t') for line in f if line.strip()]
        sentences, labels = zip(*data)
        labels = [int(label) for label in labels]
        return sentences, labels

    def _clean_sentences(self, sentences):
        """Remove punctuation and lowercase. Single place for all text cleaning."""
        return [re.sub(r'[^\w\s]', '', s).lower() for s in sentences]

    def _build_vocab(self, sentences):
        """Build token_map from a list of cleaned sentences. Only called during training."""
        vocab = set()
        for sentence in sentences:
            for word in sentence.split():
                vocab.add(word)
        sorted_vocab = sorted(list(vocab))
        token_map = {word: i for i, word in enumerate(sorted_vocab, start=2)}
        token_map['<PAD>'] = 0
        token_map['<UNK>'] = 1
        return token_map

    def _tokenize_batch(self, sentences, max_len):
        """Tokenize a list of cleaned sentences into a padded tensor [B, T]."""
        output = []
        for sentence in sentences:
            token_ids = [self.token_map.get(word, self.token_map['<UNK>']) for word in sentence.split()]
            token_ids = token_ids[:max_len]
            token_ids += [0] * (max_len - len(token_ids))
            output.append(torch.tensor(token_ids))
        return torch.stack(output)

    def tokenize_sentence(self, sentence):
        """Tokenize a single raw sentence using the stored token_map. Returns tensor [1, T]."""
        cleaned = re.sub(r'[^\w\s]', '', sentence).lower()
        token_ids = [self.token_map.get(word, self.token_map['<UNK>']) for word in cleaned.split()]
        token_ids = token_ids[:self.max_len]
        token_ids += [0] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids).unsqueeze(0)  # [1, T]


# --- Model ---
class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        embedded = self.embedding(X)            # [B, T, C]
        avg = torch.mean(embedded, dim=1)       # [B, C]
        res = self.sigmoid(self.linear(avg))
        return res


# --- Training Data ---
file_path = '/content/sample_data/sentiment_dataset_1000.txt'
dataset = DataPreprocessing(file_path)
print(f"Training data: {dataset.X.shape}, Labels: {dataset.y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    dataset.X, dataset.y, test_size=0.2, stratify=dataset.y, random_state=42
)

# --- Inference Data (loaded using the TRAINING token_map) ---
# No separate tokenize_for_inference function needed — just pass token_map to the same class
inference_file = 'sample_data/sentiment_dataset_200_neg_heavy.txt'
inf_data = DataPreprocessing(inference_file, token_map=dataset.token_map)
inf_X = inf_data.X
inf_labels = inf_data.y
print(f"Inference data: {inf_X.shape}, Labels: {inf_labels.shape}")

# --- Model Setup ---
embedding_dim = 768
model = SentimentAnalysis(len(dataset.token_map), embedding_dim)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop with metric tracking ---
epochs = 100
log_every = 10

history = {
    'epoch': [], 'train_loss': [],
    'train_acc': [], 'train_f1': [],
    'test_acc': [], 'test_f1': [],
    'inf_acc': [], 'inf_f1': [],
}

for epoch in range(epochs):
    optimizer.zero_grad()
    model_prediction = model(X_train)
    loss = loss_function(model_prediction.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % log_every == 0:
        model.eval()
        with torch.no_grad():
            train_preds = (model_prediction.squeeze() >= 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
            train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)

            test_preds = (model(X_test).squeeze() >= 0.5).float()
            test_acc = (test_preds == y_test).float().mean().item()
            test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

            inf_preds = (model(inf_X).squeeze() >= 0.5).float()
            inf_acc = (inf_preds == inf_labels).float().mean().item()
            inf_f1 = f1_score(inf_labels, inf_preds, average='weighted', zero_division=0)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        history['inf_acc'].append(inf_acc)
        history['inf_f1'].append(inf_f1)

        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.2%} F1: {train_f1:.2%} | "
              f"Test Acc: {test_acc:.2%} F1: {test_f1:.2%} | "
              f"Inf Acc: {inf_acc:.2%} F1: {inf_f1:.2%}")
        model.train()

# --- Final Evaluation ---
print("\n--- Final Test Set Evaluation ---")
with torch.no_grad():
    predictions = model(X_test).squeeze()
    predicted_labels = (predictions >= 0.5).float()
    print(f"Accuracy: {(predicted_labels == y_test).float().mean().item():.2%}")
    print(f"F1 score: {f1_score(y_test, predicted_labels, average='weighted'):.2%}")
    print(classification_report(y_test, predicted_labels, target_names=['Negative', 'Positive']))

print("\n--- Final Inference on negative-heavy dataset ---")
with torch.no_grad():
    inf_predictions = model(inf_X).squeeze()
    inf_predicted_labels = (inf_predictions >= 0.5).float()
    print(f"Accuracy: {(inf_predicted_labels == inf_labels).float().mean().item():.2%}")
    print(f"F1 score: {f1_score(inf_labels, inf_predicted_labels, average='weighted'):.2%}")
    print(classification_report(inf_labels, inf_predicted_labels, target_names=['Negative', 'Positive']))


# --- Predict sentiment for a single sentence ---
def predict_sentiment(sentence, model, dataset):
    """
    Takes a raw sentence, uses dataset's tokenize_sentence method,
    and returns the model's prediction.
    Threshold: >= 0.5 is Positive, < 0.5 is Negative.
    """
    X = dataset.tokenize_sentence(sentence)
    model.eval()
    with torch.no_grad():
        probability = model(X).squeeze().item()
    label = "Positive" if probability >= 0.5 else "Negative"
    print(f"Sentence:    {sentence}")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction:  {label} (threshold = 0.5)")
    print()

# --- Clean endpoint: user only passes a sentence ---
class SentimentPredictor:
    """Bundles the trained model and dataset so the user only needs to pass a sentence."""
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.model.eval()

    def predict(self, sentence):
        X = self.dataset.tokenize_sentence(sentence)
        with torch.no_grad():
            probability = self.model(X).squeeze().item()
        label = "Positive" if probability >= 0.5 else "Negative"
        print(f"Sentence:    {sentence}")
        print(f"Probability: {probability:.4f}")
        print(f"Prediction:  {label} (threshold = 0.5)")
        print()
        return {"sentence": sentence, "probability": probability, "label": label}

# One-time setup after training
predictor = SentimentPredictor(model, dataset)

# Example usage — just pass a sentence, nothing else:
# predictor.predict("The food was amazing and the service was great.")
# predictor.predict("This was a terrible waste of money.")
