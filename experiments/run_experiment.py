"""
Main experiment script for causal probing.
Runs experiments on multiple models with Turkish and English datasets.
"""

import random
import numpy as np
from transformers import T5EncoderModel, AutoTokenizer, ByT5Tokenizer

# Import local modules
from src.imports_and_setup import DEVICE
from src.data_generation import generate_tr_positive_dataset, generate_tr_negative_dataset
from src.hidden_states import extract_hidden_states, train_probe_per_layer
from src.control_experiments import randomization
from src.save_results import save_results_to_drive


# ============================================
# Model Configuration
# ============================================
models = [
    ("google/mt5-base",   T5EncoderModel, AutoTokenizer),   # 12 encoder layers
    ("google/mt5-large",  T5EncoderModel, AutoTokenizer),   # 24 encoder layers
    ("google/byt5-base",  T5EncoderModel, ByT5Tokenizer),   # 18 encoder layers
    ("google/byt5-large", T5EncoderModel, ByT5Tokenizer),   # 36 encoder layers
]


# ============================================
# Main Experiment
# ============================================
def run_experiment():
    """Run the main causal probing experiment."""
    
    # List to store all model results
    all_model_results = []
    random_acc_dict = {}  # Store randomization results
    
    for MODEL_NAME, model_class, tokenizer_class in models:
        print("\n" + "="*60)
        print(f"EXPERIMENT: {MODEL_NAME}")
        print("="*60)
        
        # Load model and tokenizer
        print("Loading model...")
        model = model_class.from_pretrained(MODEL_NAME).to(DEVICE)
        tokenizer = tokenizer_class.from_pretrained(MODEL_NAME)
        model.eval()
        
        # Get number of layers from config
        NUM_LAYERS = model.config.num_hidden_layers
        print(f"Number of layers: {NUM_LAYERS}")
        
        # ------------------------------------------------------------------
        # TURKISH EXPERIMENT
        # ------------------------------------------------------------------
        print("\n" + "="*50)
        print(f"EXPERIMENT 1: {MODEL_NAME} TURKISH - tur_capitals = Capital")
        print("="*50)
        
        # Create Turkish dataset
        tur_data = generate_tr_positive_dataset()
        tur_data.extend(generate_tr_negative_dataset())
        random.shuffle(tur_data)
        
        tur_sentences = [v[0] for v in tur_data]
        tur_labels = [v[1] for v in tur_data]
        tur_capital = [v[3] for v in tur_data]
        
        print(f"Total sentences: {len(tur_sentences)}")
        
        # Extract hidden states
        print("\nExtracting hidden states... (may take 1-2 minutes)")
        tur_vectors = extract_hidden_states(
            model, tokenizer, tur_sentences, tur_capital, NUM_LAYERS
        )
        
        # Train probe
        print("\nTraining probe...")
        tur_acc, tur_std = train_probe_per_layer(tur_vectors, np.array(tur_labels))
        
        # Store Turkish results
        all_model_results.append({
            'model': MODEL_NAME,
            'language': 'Turkish',
            'target': tur_capital,
            'acc': tur_acc.tolist() if isinstance(tur_acc, np.ndarray) else tur_acc,
            'std': tur_std.tolist() if isinstance(tur_std, np.ndarray) else tur_std,
            'mean_acc': np.mean(tur_acc)
        })
        
        # ---- Turkish Control Experiments ----
        print("\n" + "="*50)
        print("CONTROL EXPERIMENT 1: RANDOMIZATION (Turkish)")
        print("="*50)
        
        random_acc = randomization(tur_vectors, np.array(tur_labels))
        random_acc_dict[MODEL_NAME] = random_acc
        print("Randomization accuracy per layer:")
        for i, acc in enumerate(random_acc):
            print(f"Layer {i}: {acc:.4f}")
        
        print(f"Mean Randomization Acc: {np.mean(random_acc):.4f}")
    
    # Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save results locally
    json_path, csv_path, summary_path = save_results_to_drive(
        all_model_results,
        random_acc_dict,
    )
    
    print("\n✅ All results saved successfully!")
    print(f"📁 Location: ./results/")
    
    return all_model_results, random_acc_dict


# ============================================
# Run Experiment
# ============================================
if __name__ == "__main__":
    all_results, random_results = run_experiment()
