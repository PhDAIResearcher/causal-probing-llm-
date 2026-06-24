"""
Imports and initial setup for causal probing experiments.
"""

# ============================================
# Install required packages (uncomment if needed)
# ============================================
# !pip install transformers datasets torch matplotlib seaborn scikit-learn pandas
# !pip install transformer_lens

# ============================================
# Standard imports
# ============================================
import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer, ByT5Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Device configuration
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {DEVICE}")

# ============================================
# Constants
# ============================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
