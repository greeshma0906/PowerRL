# PowerRL


## CodeSummarizer
CodeSummarizer is a deep learning-based tool for generating natural language summaries of source code. It supports both supervised learning (via Seq2Seq with attention) and reinforcement learning (via Actor-Critic) approaches.

---
###  Model Architecture

The CodeSummarizer project implements two model architectures for automated code summarization:

#### 1. **Baseline Model: Seq2Seq with Attention**
- **Encoder**: Bidirectional LSTM
- **Decoder**: LSTM with Bahdanau-style attention mechanism
- **Loss Function**: CrossEntropyLoss (ignores PAD tokens)
- **Purpose**: Acts as a supervised baseline for comparison against reinforcement learning models.

#### 2. **Reinforcement Learning Model: Actor-Critic**
- Uses an **Actor network** to generate code summaries.
- Uses a **Critic network** to predict the expected reward (e.g., BLEU score).
- **Reward Signal**: Derived from evaluation metrics like BLEU, encouraging better long-term sequence generation.

---


###  Installation

#### 1.. Clone the repo:

```bash
   git clone https://github.com/your-username/PowerRL.git
   cd PowerRL/code_summarizer
```

#### 2. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


#### 3.Dataset Preparation

We use the dataset provided in:

> [A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation](https://arxiv.org/abs/1707.02275)

This dataset includes over 100,000 Python functions paired with corresponding docstrings.

### Steps to Prepare the Dataset:

1. **Download the dataset**:
   - Clone the author's [repository](https://github.com/sriniiyer/codenn).
   - Follow their preprocessing steps or download the processed `.json` or `.pkl` files if provided.

2. **Process the data into HDF5 format**:
   If using a custom pipeline:
   ```bash
   python scripts/preprocess_dataset.py --input data.json --output processed_data.h5
   ```

3. Ensure your final `processed_data.h5` is placed inside:
   ```
   /dataset/processed_data.h5
   ```

---




####  Training


Train the baseline Seq2Seq model:

```bash
python models/baseline_eval_model/train.py
```

Train the Actor-Critic model:

```bash
python experiments/deep_rl/train_AC.py
```


####  Evaluation

Metrics reported:
- BLEU Score
- Rouge

```bash
python experiments/deep_rl/test_AC.py
```




