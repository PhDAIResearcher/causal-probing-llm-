"""
Result saving utilities for causal probing experiments.
Saves results in JSON, CSV, and text summary formats.
"""

from datetime import datetime
import json
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Optional, Tuple


def save_results_to_drive(
    all_model_results: List[Dict[str, Any]],
    random_acc_dict: Dict[str, List[float]]
) -> Tuple[str, str, str]:
    """
    Saves all results to the results directory with multiple formats.
    Each model's randomization results are added to the output.
    
    Args:
        all_model_results: List of model result dictionaries
        random_acc_dict: Dictionary mapping model names to randomization accuracies
        
    Returns:
        Tuple of (json_path, csv_path, summary_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = './results/'
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"📁 Directory created: {save_path}")
    
    # Add randomization to each model result
    for model_result in all_model_results:
        model_name = model_result['model']
        if model_name in random_acc_dict:
            model_result['randomization'] = random_acc_dict[model_name]
        else:
            model_result['randomization'] = []
    
    # 1. Save as JSON
    results_data = {
        'timestamp': timestamp,
        'models': all_model_results
    }
    
    json_path = f"{save_path}probing_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON saved: {json_path}")
    
    # 2. Save as CSV
    rows = []
    for model_result in all_model_results:
        model_name = model_result['model']
        
        # Accuracy results
        for layer_idx, (acc, std) in enumerate(zip(
            model_result['acc'], 
            model_result['std']
        )):
            rows.append({
                'model': model_name,
                'layer': layer_idx + 1,
                'accuracy': acc,
                'std': std,
                'mean_accuracy': model_result['mean_acc'],
                'language': model_result.get('language', 'unknown')
            })
        
        # Randomization results (as separate rows)
        if 'randomization' in model_result and model_result['randomization']:
            for layer_idx, rand_acc in enumerate(model_result['randomization']):
                rows.append({
                    'model': f"{model_name}_randomization",
                    'layer': layer_idx + 1,
                    'accuracy': rand_acc,
                    'std': None,
                    'mean_accuracy': np.mean(model_result['randomization']) if model_result['randomization'] else None,
                    'language': model_result.get('language', 'unknown')
                })
    
    df = pd.DataFrame(rows)
    csv_path = f"{save_path}probing_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV saved: {csv_path}")
    
    # 3. Save summary as text
    summary_path = f"{save_path}probing_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CAUSAL PROBING EXPERIMENT RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("-"*60 + "\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("-"*60 + "\n\n")
        
        for model_result in all_model_results:
            f.write(f"\n{model_result['model']}:\n")
            f.write(f"  Mean Accuracy: {model_result['mean_acc']:.4f}\n")
            f.write(f"  Best Layer: {np.argmax(model_result['acc']) + 1} ({np.max(model_result['acc']):.4f})\n")
            f.write(f"  Layer Details:\n")
            for idx, (acc, std) in enumerate(zip(model_result['acc'], model_result['std'])):
                f.write(f"    Layer {idx+1}: {acc:.4f} ± {std:.4f}\n")
            
            # Randomization results
            if 'randomization' in model_result and model_result['randomization']:
                f.write(f"\n  Randomization Results:\n")
                f.write(f"    Mean: {np.mean(model_result['randomization']):.4f}\n")
                f.write(f"    Per Layer:\n")
                for idx, rand_acc in enumerate(model_result['randomization']):
                    f.write(f"      Layer {idx+1}: {rand_acc:.4f}\n")
        
        # Randomization summary
        f.write("\n" + "-"*60 + "\n")
        f.write("RANDOMIZATION SUMMARY\n")
        f.write("-"*60 + "\n\n")
        
        for model_result in all_model_results:
            if 'randomization' in model_result and model_result['randomization']:
                f.write(f"\n{model_result['model']} Randomization:\n")
                f.write(f"  Mean: {np.mean(model_result['randomization']):.4f}\n")
                f.write(f"  Min: {np.min(model_result['randomization']):.4f}\n")
                f.write(f"  Max: {np.max(model_result['randomization']):.4f}\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    return json_path, csv_path, summary_path
