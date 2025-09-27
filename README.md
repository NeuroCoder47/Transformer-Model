# 🤖 Transformer Model Implementation

<div align="center">

![Transformer](https://img.shields.io/badge/Model-Transformer-blue?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=2196F3&center=true&vCenter=true&random=false&width=800&lines=🚀+Attention+is+All+You+Need;✨+State-of-the-Art+Architecture;🔥+PyTorch+Implementation" alt="Typing SVG" />

</div>

---

## 🌟 Overview

This repository contains a **complete PyTorch implementation** of the Transformer architecture introduced in the groundbreaking paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│                    🏗️ Architecture                      │
├─────────────────────────────────────────────────────────┤
│  Input → Embedding → Positional Encoding               │
│    ↓                                                    │
│  Encoder Stack (N layers)                              │
│    ├── Multi-Head Attention                            │
│    ├── Add & Norm                                      │
│    ├── Feed Forward                                    │
│    └── Add & Norm                                      │
│    ↓                                                    │
│  Decoder Stack (N layers)                              │
│    ├── Masked Multi-Head Attention                     │
│    ├── Add & Norm                                      │
│    ├── Cross Multi-Head Attention                      │
│    ├── Add & Norm                                      │
│    ├── Feed Forward                                    │
│    └── Add & Norm                                      │
│    ↓                                                    │
│  Linear → Softmax → Output                             │
└─────────────────────────────────────────────────────────┘
```

</div>

## ✨ Features

- 🎯 **Multi-Head Attention** - Parallel attention mechanisms for capturing different types of relationships
- 🔄 **Positional Encoding** - Sinusoidal position embeddings for sequence order understanding
- 🏗️ **Encoder-Decoder Architecture** - Complete transformer stack with residual connections
- 🎭 **Masking Support** - Proper attention masking for causal and padding tokens
- ⚡ **Optimized Implementation** - Efficient PyTorch operations with proper tensor shapes
- 🛡️ **Bug-Free Code** - Fixed all issues from the original implementation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Transformer-Model.git
cd Transformer-Model

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib
```

### Basic Usage

```python
import torch
from model import Transformer

# Initialize model parameters
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Create the model
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=dropout
)

# Example forward pass
batch_size = 32
src_seq_len = 50
tgt_seq_len = 40

src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

output = model(src, tgt)
print(f"Output shape: {output.shape}")  # [batch_size, tgt_seq_len, tgt_vocab_size]
```

## 🏗️ Architecture Components

<details>
<summary><b>🔍 Multi-Head Attention</b></summary>

The core of the Transformer - allows the model to attend to different positions simultaneously:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # Creates multiple attention heads
        # Each head learns different relationships
```

**Key Features:**
- ✅ Parallel attention computation
- ✅ Scaled dot-product attention
- ✅ Proper head splitting and combining
- ✅ Optional masking support

</details>

<details>
<summary><b>📍 Positional Encoding</b></summary>

Since Transformers don't have built-in notion of sequence order, we add positional information:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        # Uses sinusoidal functions
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

</details>

<details>
<summary><b>🏗️ Encoder & Decoder Layers</b></summary>

**Encoder Layer:**
- Self-attention mechanism
- Position-wise feed-forward network
- Residual connections and layer normalization

**Decoder Layer:**
- Masked self-attention
- Cross-attention with encoder output
- Position-wise feed-forward network
- Residual connections and layer normalization

</details>

## 📊 Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Model dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of encoder/decoder layers |
| `d_ff` | 2048 | Feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_length` | 100 | Maximum sequence length |

## 🛠️ Training Example

```python
import torch.optim as optim
import torch.nn as nn

# Initialize model and optimizer
model = Transformer(...)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        src, tgt = batch
        
        # Forward pass
        output = model(src, tgt[:, :-1])  # Teacher forcing
        
        # Calculate loss
        loss = criterion(
            output.reshape(-1, tgt_vocab_size),
            tgt[:, 1:].reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
```

## 📈 Performance Tips

- 🚀 **Batch Size**: Use larger batch sizes for better GPU utilization
- 🎯 **Learning Rate**: Start with 0.0001 and use learning rate scheduling
- 🔥 **Warmup**: Implement learning rate warmup for better convergence
- 💾 **Memory**: Use gradient checkpointing for very large models
- ⚡ **Mixed Precision**: Enable automatic mixed precision for faster training

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Detailed explanation
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/Transformer-Model&type=Date)](https://star-history.com/#yourusername/Transformer-Model&Date)

---

<div align="center">

### 🎉 Thank you for using this Transformer implementation!

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=2000&pause=500&color=FF6B35&center=true&vCenter=true&random=false&width=600&lines=⭐+Don't+forget+to+star+this+repo!;🤝+Contributions+are+welcome!;📖+Check+out+the+documentation!" alt="Footer Typing SVG" />

**Made with ❤️ and lots of ☕**

</div>
