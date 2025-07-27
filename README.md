# 🚀 Code Comment Generator – AI-Powered Dev Tool

An AI tool that generates descriptive comments for code snippets, helping developers maintain clean, understandable, and well-documented codebases.

This project aligns with BLACKBOX AI’s mission to enhance developer productivity through AI-powered code understanding and generation tools, leveraging state-of-the-art language models to automate code documentation and improve codebase maintainability.

Inspired by the rise of AI-assisted programming tools like GitHub Copilot, this project fine-tunes a lightweight language model to automatically generate Python docstrings from source code using the [CodeSearchNet](https://huggingface.co/datasets/code_search_net) dataset.

---

## 📌 Project Overview

* **Goal**: Fine-tune a pre-trained language model (e.g., CodeT5-small) to generate Python-style documentation from source code.
* **Dataset**: A filtered subset (\~1,000 samples) of the **CodeSearchNet** dataset (Python partition).
* **Model**: [Salesforce/CodeT5-small](https://huggingface.co/Salesforce/codet5-small)
* **Frameworks**: Python, Hugging Face Transformers, Datasets
* **Platform**: Google Colab (GPU-enabled)

---

## 🎯 Objectives

* Preprocess and clean the CodeSearchNet dataset for docstring generation.
* Fine-tune CodeT5-small on code-comment pairs.
* Augment data with multiple task-specific prompts to improve generalization.
* Evaluate performance using **BLEU** and **ROUGE** metrics.
* Demonstrate generation on test samples.

---

## 📦 Setup & Requirements

* Python ≥ 3.8
* `transformers`
* `datasets`
* `evaluate`
* Google Colab or GPU-enabled environment (optional but recommended)

---

## 🧹 Data Preprocessing

### 1. **Dataset Loading**

```python
ds = load_dataset("code_search_net", "python")
```

### 2. **Filtering Criteria**

We filter examples using a function `has_valid_fields` to ensure high-quality training data:

```python
def has_valid_fields(example):
    return (
        example["func_code_string"] and
        example["func_documentation_string"] and
        len(example["func_code_string"].strip()) > 50 and
        len(example["func_documentation_string"].strip()) > 10 and
        not example["func_documentation_string"].startswith("TODO") and
        example["func_documentation_string"].count('\n') < 5
    )
```

✅ This helps:

* Eliminate incomplete or placeholder examples.
* Ensure minimum code and documentation length.
* Reduce training noise.

### 3. **Cleaning & Formatting**

* Strip and clean docstring quotes (`'''`, `"""`).
* Format inputs as `"Generate docstring: <code>"` and outputs as the cleaned documentation.

```python
def process(example):
    doc = example["func_documentation_string"].strip().replace('"""', '').replace("'''", '')
    return {
        "input_text": "Generate docstring: " + example["func_code_string"],
        "target_text": doc
    }
```

---

## 📈 Data Augmentation

To improve robustness, we manually generate **prompt variations**:

* Prompts like:

  * `"Explain this function:"`
  * `"Document this code:"`
  * `"What does this do?"`

Each code snippet is paired with multiple prompts but the same docstring.

📌 Outcome:

* **Original Size**: `N`
* **Augmented Size**: `~4N`

This increases model generalization without external data.

---

## ✂️ Tokenization

### Key Parameters:

| Feature       | Value         |
| ------------- | ------------- |
| Input Length  | 768 tokens    |
| Output Length | 256 tokens    |
| Tokenizer     | CodeT5’s BBPE |
| Padding       | max\_length   |
| Truncation    | right         |

```python
tokenized = tokenizer(
    example["input_text"],
    padding="max_length",
    truncation=True,
    max_length=768
)

with tokenizer.as_target_tokenizer():
    labels = tokenizer(
        example["target_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
```

---

## 🧪 Evaluation Metrics

A custom `compute_metrics` function evaluates model outputs using:

* **BLEU**: Precision of predicted n-grams vs reference.
* **ROUGE-1/2/L**: Measures overlap and sequence quality.

Also includes:

* Automatic decoding of predictions.
* Sample printouts and error handling.

---

## 🧠 Model Training

### Configuration:

| Parameter             | Value        |
| --------------------- | ------------ |
| Model                 | CodeT5-small |
| Train Size            | 1,000        |
| Eval Size             | 200          |
| Epochs                | 8            |
| Learning Rate         | 3e-4         |
| Batch Size            | 4            |
| Eval Steps            | 200          |
| Save Steps            | 200          |
| Mixed Precision       | FP16         |
| Gradient Accumulation | 2            |
| Generation Length     | 256          |
| Beam Search           | 6 beams      |

> ✅ The `Trainer` from Hugging Face manages the training loop, evaluation, checkpointing, and metrics.

---

## 📊 Results

| Step | Train Loss | Val Loss | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ---- | ---------- | -------- | ---- | ------- | ------- | ------- |
| 200  | 0.0113     | 0.0031   | 0.88 | 0.98    | 0.98    | 0.98    |
| 400  | 0.0045     | 0.0021   | 0.98 | 0.99    | 0.98    | 0.99    |
| 600  | 0.0040     | 0.0014   | 0.96 | 0.99    | 0.98    | 0.99    |
| 800  | 0.0010     | 0.0010   | 0.99 | 0.99    | 0.99    | 0.99    |
| 1000 | 0.0006     | 0.0009   | 0.99 | 0.99    | 0.99    | 0.99    |

---

## 📂 Deliverables

* 🧪 Jupyter notebook.
* 📎 Sample test inputs and generated outputs. (in the last cell of the notebook)
* 📈 Evaluation metrics (BLEU, ROUGE).
* 📄 This README.

---

## 🛠️ Skills Demonstrated

* LLM fine-tuning (CodeT5)
* Dataset preprocessing and augmentation
* Text generation evaluation
* Efficient prototyping in Colab

---

## 📬 Contact

For questions, suggestions, or contributions, feel free to open an issue or reach out.

NOTE : the metrics of training and the testing on code snippets are in the notebook

---
