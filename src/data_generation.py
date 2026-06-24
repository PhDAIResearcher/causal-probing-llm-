"""
Data generation functions for causal probing experiments.
Generates positive, negative, and control datasets in Turkish and English.
"""

import random
from typing import List, Tuple, Dict, Any

# Import templates and utilities from data_templates
from src.data_templates import (
    capitals_tr, capitals_en,
    tr_positive, tr_negative, tr_control,
    en_positive, en_indirect_templates, en_control,
    make_genitive, add_locative_suffix, add_copula_suffix
)

# ============================================
# Turkish Dataset Generation
# ============================================

def generate_tr_positive_dataset() -> List[Tuple[str, int, str, str, str, str]]:
    """
    Generate Turkish positive (capital) dataset.
    
    Returns:
        List of tuples: (sentence, label, language, capital, country, category)
    """
    dataset = []
    random.seed(42)
    
    for country, capital in capitals_tr.items():
        for template in tr_positive:
            # Determine capital form based on template
            if template.strip().startswith("{capital}"):
                capital_form = capital
            else:
                capital_form = add_copula_suffix(capital)  # Ankara'dır
            
            sentence = template.format(
                country=make_genitive(country),
                capital=capital_form,
                capital_suffix=add_locative_suffix(capital)
            )
            dataset.append((sentence, 1, "TR", capital, country, "positive"))
    
    random.shuffle(dataset)
    print(f"Turkish positive dataset created: {len(dataset)} sentences")
    return dataset


def generate_tr_negative_dataset() -> List[Tuple[str, int, str, str, str, str]]:
    """
    Generate Turkish negative (non-capital) dataset.
    
    Returns:
        List of tuples: (sentence, label, language, capital, country, category)
    """
    dataset = []
    random.seed(42)
    
    for country, capital in capitals_tr.items():
        for template in tr_negative:
            if template.strip().startswith("{capital}"):
                capital_form = capital
            else:
                capital_form = add_copula_suffix(capital)  # Ankara'dır
            
            sentence = template.format(
                country=make_genitive(country),
                capital=capital_form,
                capital_suffix=add_locative_suffix(capital)
            )
            dataset.append((sentence, 0, "TR", capital, country, "negative"))
    
    random.shuffle(dataset)
    print(f"Turkish negative dataset created: {len(dataset)} sentences")
    return dataset


def generate_tr_control_dataset() -> List[Tuple[str, int, str, str, str, str]]:
    """
    Generate Turkish control (co-occurrence) dataset.
    
    Returns:
        List of tuples: (sentence, label, language, city, country, category)
    """
    dataset = []
    random.seed(42)
    
    for country, capital in capitals_tr.items():
        for template in tr_control:
            sentence = template.format(
                country=make_genitive(country),
                city=capital
            )
            # Control sentences are labeled as 0 (negative)
            dataset.append((sentence, 0, "TR", capital, country, "control"))
    
    random.shuffle(dataset)
    print(f"Turkish control dataset created: {len(dataset)} sentences")
    return dataset


# ============================================
# English Dataset Generation
# ============================================

def generate_en_positive_dataset() -> List[Tuple[str, int, str, str, str, str]]:
    """
    Generate English positive (capital) dataset.
    
    Returns:
        List of tuples: (sentence, label, language, capital, country, category)
    """
    dataset = []
    random.seed(42)
    
    for country, capital in capitals_en.items():
        for template in en_positive:
            sentence = template.format(
                country=country,
                capital=capital,
                capital_suffix=capital
            )
            dataset.append((sentence, 1, "EN", capital, country, "positive"))
    
    random.shuffle(dataset)
    print(f"English positive dataset created: {len(dataset)} sentences")
    return dataset


def generate_en_indirect_dataset() -> List[Tuple[str, int, str, str, str, str]]:
    """
    Generate English indirect dataset.
    
    Returns:
        List of tuples: (sentence, label, language, capital, country, category)
    """
    dataset = []
    random.seed(42)
    
    for country, capital in capitals_en.items():
        for template in en_indirect_templates:
            sentence = template.format(
                country=country,
                capital=capital,
                capital_suffix=capital
            )
            dataset.append((sentence, 1, "EN", capital, country, "indirect"))
    
    random.shuffle(dataset)
    print(f"English indirect dataset created: {len(dataset)} sentences")
    return dataset


def generate_en_control_dataset() -> List[Tuple[str, int, str, str, str, str]]:
    """
    Generate English control (co-occurrence) dataset.
    
    Returns:
        List of tuples: (sentence, label, language, city, country, category)
    """
    dataset = []
    random.seed(42)
    
    for country, capital in capitals_en.items():
        for template in en_control:
            sentence = template.format(
                country=country,
                city=capital
            )
            dataset.append((sentence, 0, "EN", capital, country, "control"))
    
    random.shuffle(dataset)
    print(f"English control dataset created: {len(dataset)} sentences")
    return dataset


# ============================================
# Combined Dataset Generators
# ============================================

def generate_all_turkish_datasets() -> Dict[str, List[Tuple[str, int, str, str, str, str]]]:
    """
    Generate all Turkish datasets.
    
    Returns:
        Dictionary with keys: 'positive', 'negative', 'control'
    """
    return {
        'positive': generate_tr_positive_dataset(),
        'negative': generate_tr_negative_dataset(),
        'control': generate_tr_control_dataset()
    }


def generate_all_english_datasets() -> Dict[str, List[Tuple[str, int, str, str, str, str]]]:
    """
    Generate all English datasets.
    
    Returns:
        Dictionary with keys: 'positive', 'indirect', 'control'
    """
    return {
        'positive': generate_en_positive_dataset(),
        'indirect': generate_en_indirect_dataset(),
        'control': generate_en_control_dataset()
    }


def generate_full_dataset() -> Dict[str, Dict[str, List[Tuple[str, int, str, str, str, str]]]]:
    """
    Generate all datasets for all languages.
    
    Returns:
        Dictionary with keys: 'TR', 'EN'
    """
    return {
        'TR': generate_all_turkish_datasets(),
        'EN': generate_all_english_datasets()
    }


# ============================================
# Dataset Utilities
# ============================================

def get_dataset_statistics(dataset: List[Tuple]) -> Dict[str, Any]:
    """
    Get statistics for a dataset.
    
    Args:
        dataset: List of tuples (sentence, label, language, ...)
        
    Returns:
        Dictionary with statistics
    """
    labels = [item[1] for item in dataset]
    languages = [item[2] for item in dataset]
    categories = [item[5] for item in dataset]
    
    return {
        'total_samples': len(dataset),
        'positive_samples': sum(labels),
        'negative_samples': len(labels) - sum(labels),
        'languages': list(set(languages)),
        'categories': list(set(categories)),
        'category_counts': {cat: categories.count(cat) for cat in set(categories)}
    }


def sample_dataset(
    dataset: List[Tuple],
    n_samples: int,
    balanced: bool = False
) -> List[Tuple]:
    """
    Sample from a dataset.
    
    Args:
        dataset: List of tuples
        n_samples: Number of samples to take
        balanced: If True, balance positive and negative samples
        
    Returns:
        Sampled dataset
    """
    random.seed(42)
    
    if not balanced:
        return random.sample(dataset, min(n_samples, len(dataset)))
    
    # Balanced sampling
    positives = [item for item in dataset if item[1] == 1]
    negatives = [item for item in dataset if item[1] == 0]
    
    n_per_class = n_samples // 2
    sampled_pos = random.sample(positives, min(n_per_class, len(positives)))
    sampled_neg = random.sample(negatives, min(n_per_class, len(negatives)))
    
    sampled = sampled_pos + sampled_neg
    random.shuffle(sampled)
    
    return sampled


def print_dataset_summary(datasets: Dict[str, List[Tuple]]) -> None:
    """
    Print summary of multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset name to dataset list
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for name, dataset in datasets.items():
        stats = get_dataset_statistics(dataset)
        print(f"\n📊 {name.upper()}:")
        print(f"  Total: {stats['total_samples']}")
        print(f"  Positive: {stats['positive_samples']}")
        print(f"  Negative: {stats['negative_samples']}")
        print(f"  Categories: {stats['categories']}")
        print(f"  Category counts: {stats['category_counts']}")


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Generate datasets
    print("Generating datasets...\n")
    
    tr_positive = generate_tr_positive_dataset()
    tr_negative = generate_tr_negative_dataset()
    tr_control = generate_tr_control_dataset()
    
    en_positive = generate_en_positive_dataset()
    en_indirect = generate_en_indirect_dataset()
    en_control = generate_en_control_dataset()
    
    # Print summaries
    print_dataset_summary({
        'turkish_positive': tr_positive,
        'turkish_negative': tr_negative,
        'turkish_control': tr_control,
        'english_positive': en_positive,
        'english_indirect': en_indirect,
        'english_control': en_control
    })
    
    # Show sample sentences
    print("\n" + "="*60)
    print("SAMPLE SENTENCES")
    print("="*60)
    
    print("\n🇹🇷 Turkish Positive:")
    for i in range(3):
        print(f"  {tr_positive[i][0]}")
    
    print("\n🇹🇷 Turkish Negative:")
    for i in range(3):
        print(f"  {tr_negative[i][0]}")
    
    print("\n🇬🇧 English Positive:")
    for i in range(3):
        print(f"  {en_positive[i][0]}")
    
    print("\n🇬🇧 English Indirect:")
    for i in range(3):
        print(f"  {en_indirect[i][0]}")
