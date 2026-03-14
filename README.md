# ChatMed-VHI

ChatMed-VHI is a biomedical information extraction framework for mining Virus–Host–Interaction (VHI) entities from multi-source biomedical evidence.

This project explores instruction-tuning and prompt learning approaches for extracting virus, host, and interaction entities from biomedical literature and datasets.

The repository provides:

- Data preprocessing for VHI datasets
- BERT-based fine-tuning for entity extraction
- GPT-based instruction prompting
- Evaluation metrics for biomedical information extraction
---

## Project Structure

ChatMed-VHI/

│

├── BERT_fine_tune/
│   ├── Bert_finetune.ipynb
│   ├── qa_BIOtoken.py
│   └── utils_qa.py
│

├── GPT_fine_tune/
│   ├── GPT_fine_tune.ipynb
│   └── GPT_zero_five_prompt.ipynb
│

├── Metrics/
│   └── Metrics.ipynb
│

├── data_process/
│   └── data_process.ipynb
│

├── fine_tune_data/
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   ├── train10%_finetune.jsonl
│   ├── train25%_finetune.jsonl
│   ├── train50%_finetune.jsonl
│   └── train100%_finetune.jsonl
│

├── README.md
└── LICENSE

---

## Installation

Clone the repository:

git clone https://github.com/benkyusimasu/ChatMed-VHI.git
cd ChatMed-VHI

Create a virtual environment (recommended):

conda create -n chatmed python=3.9
conda activate chatmed

Install dependencies:

pip install -r requirements.txt

---

## Requirements

Main dependencies:

torch
transformers
datasets
numpy
pandas
scikit-learn
tqdm
evaluate

---

## Dataset Format

The dataset follows the HuggingFace Question Answering format.

Example:
{

"id": "msxx4oe8zok46m3n0xzqphm0",

"question": "What is the name of the virus whose protein interacts with host factors/proteins", 

"context": "To further explore the effects of different MOIs of PR8 virus on hnRNPH1 expression, the western blot and qPCR were used to detect the protein and mRNA levels of hnRNPH1. The results showed that there were no significant difference in the protein and mRNA levels of hnRNPH1 after PR8 infection at different MOIs (MOI = 0.1, 1, 10) (Figure S1B and C). Besides, the overexpression and interference of hnRNPH1 had similar regulatory effects on viral replication in cells infected with different MOIs of PR8 (Figure S1D and E), suggesting that the interaction between hnRNPH1 and NS1 may be not affected by the amount of viral infection. ", 

"answers": {"answer_start": [52], "text": ["PR8"]}

}


Field explanation:

context — virus host interaction text context
question — question related to the context
answers — answer span and start position

---

## Workflow

The workflow of ChatMed-VHI consists of four stages:

1. Data preprocessing
2. BERT fine-tuning
3. GPT prompt experiments
4. Model evaluation

---

## Data Preprocessing

Run the preprocessing notebook:

data_process/data_process.ipynb

This step prepares the dataset for training and evaluation.

---
## BERT Fine-tuning

Run:

BERT_fine_tune/Bert_finetune.ipynb

or execute the training script:

python BERT_fine_tune/qa_BIOtoken.py

The model is based on the HuggingFace Transformers Question Answering framework.

---

## GPT Prompt Experiments

Prompt-based experiments are located in:

GPT_fine_tune/

Includes:

• zero-shot prompting
• few-shot prompting

Example:

GPT_fine_tune/GPT_fine_tune.ipynb

---

## Model Evaluation

Evaluation metrics are implemented in:

Metrics/Metrics.ipynb

Metrics include:

• Precision
• Recall
• F1 Score
• Exact Match
• Partial Match

---

## Experiments

The repository includes experiments using different training data proportions:

10% training data
25% training data
50% training data
100% training data

These experiments evaluate model performance under low-resource settings.

---

## Applications

Possible applications include:

• Host pathogen interaction information extraction
• Virus host interaction information extraction
• Medical information extraction

---
## Paper

Mining Virus, Host, and Interaction Entities from Multi-Source Evidence via Instruction-tuning
https://doi.org/10.1101/2025.09.02.673691

## Citation

If you use this repository in your research, please cite:

@misc{chatmedvhi2026,
title={ChatMed-VHI: Mining Virus, Host, and Interaction Entities from Multi-Source Evidence via Instruction-tuning},
author={Zhang Zheng},
year={2026},
url={https://github.com/benkyusimasu/ChatMed-VHI}
}

---

## License

This project is licensed under the MIT License.
