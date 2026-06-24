This repository presents a comprehensive framework for causal probing and mechanistic interpretability of transformer-based large language models. Unlike traditional correlational probing methods that merely identify statistical associations, our approach employs causal interventions (activation patching, randomized ablation, and counterfactual manipulations) to establish genuine causal relationships between internal representations and model outputs.

Research Questions
Where is specific linguistic/semantic information encoded across model layers?

How does information flow through the transformer architecture?

Which layers are causally responsible for particular tasks (e.g., tense detection, subject-verb agreement, logical reasoning)?

Can we intervene on internal representations to control or modify model behavior?

Methodology
Activation Patching: Interchange latent representations between different inputs to test causal effects.

Randomized Ablation: Corrupt specific layer outputs with Gaussian noise or shuffled representations to assess their contribution.

Counterfactual Interventions: Manipulate specific token embeddings to observe downstream effects.

Layer-wise Analysis: Systematic evaluation across all transformer layers to build a causal circuit map.

Key Features
🧪 Modular framework for designing causal probing experiments

🔬 Support for multiple intervention types (patch, swap, ablate, add noise)

📊 Built-in visualization tools for layer contribution heatmaps

🤖 Compatible with HuggingFace transformers (GPT-2, LLaMA, Mistral, etc.)

⚡ Efficient batch processing for large-scale analysis

Technical Stack
Python 3.10+

PyTorch + HuggingFace Transformers

Pyvene for causal interventions

Matplotlib / Seaborn for visualization

Weights & Biases for experiment tracking (optional)

Applications
🧠 Mechanistic Interpretability: Understanding how LLMs process information internally

🛡️ Safety & Robustness: Identifying vulnerable layers for adversarial attacks

✂️ Model Compression: Pruning non-causal layers to optimize efficiency

🎯 Controllable Generation: Steering model outputs by targeted layer interventions
