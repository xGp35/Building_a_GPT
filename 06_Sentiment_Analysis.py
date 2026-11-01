class DataPreprocessing:
    def __init__(self, file_path: str, max_len: int = 10):
        # Calling the 2 methods defined inside this class.

        sentences, labels = self.get_dataset(file_path)
        self.token_map, self.X = self.tokenize_dataset(sentences, max_len)

        # This self.token_map is a object instance level attribute. This is diffrent from the local variable token_map defined inside the tokenize_dataset. That method returns that local variable and here we capture that local variable as an object attribute.

        self.y = torch.tensor(labels, dtype=torch.float32)
        # Don't write self.token_map = {}, here. This is already being defined in the call to self.tokenize_dataset(). Having this here will replce the value created by tokenize_dataset with an empty dictionary.

    def get_dataset(self, file_path):
        # parse file, return (list_of_sentences, list_of_labels)
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            data = [line.strip().split('\t') for line in f if line.strip()]
            # line.strip() gives one string per row of f. Then, .split() is applied on each string. This gives a list for each string.
            # So from list of strings, we go to list of lists. Where the inner list has 2 elements, a sentence and a label. 
        sentences, labels = zip(*data) # sentences and labels are tuples of length 69 now.
        sentences = [re.sub(r'[^\w\s]', '', s).lower() for s in sentences] #Remove all punctuations from sentences and convert everything to lowercase.
        labels = [int(label) for label in labels] # labels is converted into a list
        return sentences, labels

    def tokenize_dataset(self, sentences, max_len):
        # your existing logic, but also return the vocab
        # return token_map, padded_tensor

        # All sentences is list of tuple. Each tuple has a sentence and a label.
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        vocab = set()
        splitted_sentences = [sentence.split() for sentence in sentences] # Create a list of lists from the sentences list. Each element of inner list is now a word.
        for splitted_sentence in splitted_sentences:
            for word in splitted_sentence:
                vocab.add(word)
        sorted_vocab = sorted(list(vocab))
        token_map = {word: i for i, word in enumerate(sorted_vocab, start = 2)}
        token_map['<PAD>'] = 0 # ) for padding
        token_map["<UNK>"] = 1 # For unknown words.
        # Note: this token_map is a local variable different from the class attribute self.token_map. That will contian the value of this local variable once we return this local variable in the return statement.

        # 2. Encode each sentence by replacing words with their IDs
        output = []
        for splitted_sentence in splitted_sentences:
            tokenized_sentence = []
            for word in splitted_sentence:
                tokenized_sentence.append(token_map[word])
            # Truncate if too long
            tokenized_sentence = tokenized_sentence[:max_len] # max_len is T
            # Pad if too short
            tokenized_sentence += [0] * (max_len - len(tokenized_sentence))
            output.append(torch.tensor(tokenized_sentence))
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # 'output' now a python list of 1D tensors. To make it a proper 2D tensor, we use torch.stack. This gives us [B, T], ie, 69x10
        return token_map, torch.stack(output)

class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # say vocab size = 444, then this is 444 x768
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        # In pytorch model contructor we don't include information about the batch size but we include batch size information in forward method.
    def forward(self, X):
        # X shape: [B, T] (Batch size x Sequence length)
        embedded = self.embedding(X) # This gives a vector of dim B x T x C
        avg = torch.mean(embedded, dim = 1) # This gives a vector of size B x C
        res = self.sigmoid(self.linear(avg))
        return res

# Run the batch through SentimentAnalysis model
embedding_dim = 768
model = SentimentAnalysis(len(dataset.token_map), embedding_dim)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    model_prediction = model(X_train)

    loss = loss_function(model_prediction.squeeze(), y_train) # The .squeeze() turns the [60, 1] output into [60] to match y_train's shape.
    loss.backward()
    optimizer.step()


# Model Evaluation
# Turn off gradient computation (as we are not training), and convert the predictions to 0/1 labels to compare against y_test.

with torch.no_grad():
    predictions = model(X_test).squeeze()
    predicted_labels = (predictions >= 0.5).float()
    accuracy = (predicted_labels == y_test).float().mean()
    print(f"Test accuracy: {accuracy.item():.2%}")
    print(f"F1 score: {f1_score(y_test, predicted_labels, average='weighted'):.2%}")
    print(classification_report(y_test, predicted_labels, target_names=['Negative', 'Positive']))

# --- Inference on unseen data (negative-heavy dataset) ---
# This dataset was NOT used for training. The model has never seen these sentences.
# We tokenize using the TRAINING token_map so that words map to the same IDs the model learned.
# Words not in the training vocabulary get mapped to <UNK> (ID = 1).

def tokenize_for_inference(sentences, token_map, max_len=10):
    """Tokenize new sentences using the training vocabulary's token_map."""
    output = []
    for sentence in sentences:
        token_ids = [token_map.get(word, token_map["<UNK>"]) for word in sentence.split()]
        token_ids = token_ids[:max_len]
        token_ids += [0] * (max_len - len(token_ids))
        output.append(torch.tensor(token_ids))
    return torch.stack(output)

# Load the inference dataset (75% negative, 25% positive)
inference_file = 'data/sentiment_dataset_200_neg_heavy.txt'
with open(inference_file, 'r', encoding='utf-8') as f:
    next(f)  # Skip header
    data = [line.strip().split('\t') for line in f if line.strip()]
inf_sentences, inf_labels = zip(*data)
inf_sentences = [re.sub(r'[^\w\s]', '', s).lower() for s in inf_sentences]
inf_labels = torch.tensor([int(label) for label in inf_labels], dtype=torch.float32)

# Tokenize using the TRAINING token_map — not building a new vocabulary
inf_X = tokenize_for_inference(inf_sentences, dataset.token_map)

# Run inference
with torch.no_grad():
    inf_predictions = model(inf_X).squeeze()
    inf_predicted_labels = (inf_predictions >= 0.5).float()
    inf_accuracy = (inf_predicted_labels == inf_labels).float().mean()
    print(f"\n--- Inference on negative-heavy dataset (200 sentences) ---")
    print(f"Accuracy: {inf_accuracy.item():.2%}")
    print(f"F1 score: {f1_score(inf_labels, inf_predicted_labels, average='weighted'):.2%}")
    print(classification_report(inf_labels, inf_predicted_labels, target_names=['Negative', 'Positive']))