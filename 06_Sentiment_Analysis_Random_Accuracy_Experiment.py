import re
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# --- Classes copied from 06_Sentiment_Analysis.py ---
# (Can't import directly because that file has executable code at module level and no imports)

class DataPreprocessing:
    def __init__(self, file_path: str, max_len: int = 10):
        sentences, labels = self.get_dataset(file_path)
        self.token_map, self.X = self.tokenize_dataset(sentences, max_len)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def get_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            data = [line.strip().split('\t') for line in f if line.strip()]
        sentences, labels = zip(*data)
        sentences = [re.sub(r'[^\w\s]', '', s).lower() for s in sentences]
        labels = [int(label) for label in labels]
        return sentences, labels

    def tokenize_dataset(self, sentences, max_len):
        vocab = set()
        splitted_sentences = [sentence.split() for sentence in sentences]
        for splitted_sentence in splitted_sentences:
            for word in splitted_sentence:
                vocab.add(word)
        sorted_vocab = sorted(list(vocab))
        token_map = {word: i for i, word in enumerate(sorted_vocab, start=2)}
        token_map['<PAD>'] = 0
        token_map['<UNK>'] = 1

        output = []
        for splitted_sentence in splitted_sentences:
            tokenized_sentence = []
            for word in splitted_sentence:
                tokenized_sentence.append(token_map[word])
            tokenized_sentence = tokenized_sentence[:max_len]
            tokenized_sentence += [0] * (max_len - len(tokenized_sentence))
            output.append(torch.tensor(tokenized_sentence))
        return token_map, torch.stack(output)


class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        embedded = self.embedding(X)
        avg = torch.mean(embedded, dim=1)
        res = self.sigmoid(self.linear(avg))
        return res


# --- Data Setup ---
file_path = 'data/sentiment_dataset_69.txt'
dataset = DataPreprocessing(file_path)

# +1 for padding index 0 which is not a key count in token_map but is a valid embedding index
vocab_size = len(dataset.token_map) + 1

X_train, X_test, y_train, y_test = train_test_split(
    dataset.X, dataset.y, test_size=9, stratify=dataset.y, random_state=42
)

print(f"Dataset: {len(dataset.y)} total, {len(y_train)} train, {len(y_test)} test")
print(f"Vocab size: {vocab_size}")
print(f"Test set labels: {y_test.tolist()}")
print()

# --- Experiment: Untrained model accuracy over 100 trials ---
num_trials = 100
embedding_dim = 768
accuracies = []

for i in range(num_trials):
    untrained_model = SentimentAnalysis(vocab_size, embedding_dim)
    with torch.no_grad():
        predictions = untrained_model(X_test).squeeze()
        predicted_labels = (predictions >= 0.5).float()
        acc = (predicted_labels == y_test).float().mean().item()
        accuracies.append(acc)

accuracies_tensor = torch.tensor(accuracies)
print(f"--- Results over {num_trials} trials ---")
print(f"Mean accuracy: {accuracies_tensor.mean().item():.2%}")
print(f"Std deviation: {accuracies_tensor.std().item():.2%}")
print(f"Min accuracy:  {accuracies_tensor.min().item():.2%}")
print(f"Max accuracy:  {accuracies_tensor.max().item():.2%}")
