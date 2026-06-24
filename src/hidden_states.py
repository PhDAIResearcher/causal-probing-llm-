"""
Hidden state extraction and probe classifier training functions.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_hidden_states(model, tokenizer, sentences, target_tokens, num_layers):
    """
    Returns:
        all_vectors: Tensor of shape (num_layers, num_sentences, 768)
                     Each vector is the average of all target tokens in the sentence
    """
    print(f"Target tokens: {target_tokens}")

    all_vectors = []

    # Analyze each target token first
    token_analysis = {}
    for token in target_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        token_texts = tokenizer.convert_ids_to_tokens(token_ids)

        token_analysis[token] = {
            'ids': token_ids,
            'texts': token_texts,
            'is_single': len(token_ids) == 1,
            'length': len(token_ids)
        }
        """
        print(f"\n📌 Token: '{token}'")
        print(f"   ID'ler: {token_ids}")
        print(f"   Metinler: {token_texts}")
        print(f"   Durum: {'TEK' if token_analysis[token]['is_single'] else f'PARÇALI ({len(token_ids)} parça)'}")
        """
    for sentence in sentences:
        #print(f"\n📝 Cümle: {sentence}")

        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(DEVICE)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        #print(f"   Token'lar: {tokens}")

        # Capture layer outputs with hooks
        layer_outputs = []

        def hook_fn(module, input, output):
            layer_outputs.append(output[0].detach().cpu())

        # Add hooks to each layer
        hooks = []
        for layer in model.encoder.block:
            hooks.append(layer.register_forward_hook(hook_fn))

        # Run the model
        with torch.no_grad():
            _ = model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Find indices for each target token in this sentence
        sentence_indices = []

        for token in target_tokens:
            token_ids = token_analysis[token]['ids']
            token_texts = token_analysis[token]['texts']

            target_idx = None

            # Strategy 1: If single token, search for exact match
            if len(token_ids) == 1:
                token_id = token_ids[0]
                matches = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    target_idx = matches[0].item()
                    #print(f"   ✅ '{token}' -> pozisyon: {target_idx}")

            # Strategy 2: If multi-token, search for the sequence
            elif len(token_ids) > 1:
                token_ids_tensor = torch.tensor(token_ids).to(DEVICE)
                for i in range(len(input_ids[0]) - len(token_ids) + 1):
                     if torch.all(input_ids[0][i:i+len(token_ids)] == token_ids_tensor):
                        target_idx = i
                        #print(f"   ✅ '{token}' (parçalı) -> pozisyon: {target_idx}")
                        break

            # Strategy 3: If not found, try to find by token text
            if target_idx is None:
                for i, token_text in enumerate(tokens):
                    for target_text in token_texts:
                        if target_text in token_text or token_text in target_text:
                            target_idx = i
                            #print(f"   ⚠️ '{token}' metin eşleşmesi -> pozisyon: {target_idx}")
                            break
                    if target_idx is not None:
                        break

            # Strategy 4: Default position
            if target_idx is None:
                target_idx = 1  # Usually not [CLS]
                #print(f"   ❌ '{token}' bulunamadı, varsayılan: {target_idx}")

            sentence_indices.append(target_idx)

        # Get the target token vectors for each layer
        sentence_vectors = []
        for layer_idx, hidden_state in enumerate(layer_outputs[:num_layers]):
            # hidden_state shape: (1, num_tokens, 768)
            # Collect vectors for all target tokens
            token_vectors = []
            for idx in sentence_indices:
                target_vector = hidden_state[0, idx, :].numpy()
                token_vectors.append(target_vector)

            # Average the vectors of all target tokens in this sentence
            # Shape: (768,)
            averaged_vector = np.mean(token_vectors, axis=0)
            sentence_vectors.append(averaged_vector)

        # Stack all layers for this sentence
        # Shape: (num_layers, 768)
        sentence_vectors = np.stack(sentence_vectors)
        all_vectors.append(sentence_vectors)

    # Convert to (num_layers, num_sentences, 768) format
    all_vectors = np.array(all_vectors).transpose(1, 0, 2)

    return all_vectors


def train_probe_per_layer(vectors, labels, cv_folds=None):
    """
    Trains a logistic regression classifier for each layer separately.
    If cv_folds is None, uses train/test split; otherwise performs cross-validation.
    """

    num_layers = vectors.shape[0]
    accuracies = []
    stds = []

    # Check the number of samples in each class
    unique, counts = np.unique(labels, return_counts=True)
    min_class_size = min(counts)

    # Automatically set number of CV folds
    if cv_folds is None:
        if min_class_size >= 5:
            cv_folds = 5
        elif min_class_size >= 3:
            cv_folds = 3
        else:
            cv_folds = 2

    print(f"Min class size: {min_class_size}, Using CV folds: {cv_folds}")

    for layer in range(num_layers):
        X = vectors[layer]

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        )

        if cv_folds > 1 and min_class_size >= cv_folds:
            # Cross-validation possible
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy')
            accuracies.append(scores.mean())
            stds.append(scores.std())
        else:
            # Cross-validation not possible, use train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.3, stratify=labels, random_state=42
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            stds.append(0)  # Single measurement, no std

        print(f"Layer {layer+1}: Accuracy = {accuracies[-1]:.3f} ± {stds[-1]:.3f}")

    return accuracies, stds
