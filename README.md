# ChatMed-VHI

ChatMed-VHI is a medical NLP model designed for information extraction in Virus Host Interaction (VHI).

This repository provides training pipelines for BERT and GPT based medical models.

---

# Project Structure

```
ChatMed-VHI
│
├── BERT_fine_tune        # BERT fine-tuning scripts
├── GPT_fine_tune         # GPT fine-tuning scripts
├── Metrics               # evaluation metrics
├── data_process          # dataset preprocessing
├── fine_tune_data        # training datasets
```

---

# Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Quick Start

## 1 Data Processing

```
python data_process/preprocess.py
```

---

## 2 BERT Fine-tuning

```
python BERT_fine_tune/train.py
```

---

## 3 GPT Fine-tuning

```
python GPT_fine_tune/train.py
```

---

# Dataset Format

Example training data format:

```json
{
 "text": "Patient has severe tooth pain",
 "entities": [
   {"type": "Symptom", "value": "tooth pain"}
 ]
}
```

Fields:

| Field | Description |
|------|-------------|
| text | medical dialogue text |
| entities | labeled medical entities |

---

# Evaluation

Evaluation scripts are provided in:

```
Metrics/
```

Example:

```
python Metrics/evaluate.py
```

Metrics include:

- Precision
- Recall
- F1 Score

---

# Citation

If you use this project in your research, please cite:

```
@misc{chatmedvhi2026,
  title={ChatMed-VHI: Medical Dialogue Understanding Model},
  author={Zhang Zheng},
  year={2026},
  url={https://github.com/benkyusimasu/ChatMed-VHI}
}
```

---

# License

MIT License
