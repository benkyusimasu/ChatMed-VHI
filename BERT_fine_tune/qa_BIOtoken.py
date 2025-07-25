#!/usr/bin/env python
# coding=utf-8
"""
修改后的问答任务 fine-tuning 脚本，适用于 Google Colab 运行，并在评估时计算如下 6 个指标：
  partial_precision, partial_recall, partial_f1,
  exact_precision, exact_recall, exact_f1

在 Colab 中运行前，请确保安装以下依赖：
    !pip install transformers datasets
"""

import logging
import os
import sys
import re
import string
import collections
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

# 版本检查
check_min_version("4.9.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

# -------------------- 模型与数据参数定义 --------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch, tag or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Use the token generated from transformers-cli login for private models."},
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (json or csv)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "The evaluation data file (json or csv)."})
    test_file: Optional[str] = field(default=None, metadata={"help": "The test data file (json or csv)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Number of processes for preprocessing."})
    do_not_use_token_type_ids: bool = field(default=False, metadata={"help": "If true, do not use token_type_ids."})
    max_seq_length: int = field(default=384, metadata={"help": "Maximum input sequence length after tokenization."})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad all samples to max_seq_length."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate training examples for debugging."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate evaluation examples for debugging."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Truncate prediction examples for debugging."})
    version_2_with_negative: bool = field(default=False, metadata={"help": "If true, some examples do not have an answer."})
    null_score_diff_threshold: float = field(default=0.0, metadata={"help": "Threshold for selecting null answer."})
    doc_stride: int = field(default=128, metadata={"help": "Stride when splitting long documents."})
    n_best_size: int = field(default=20, metadata={"help": "Total number of n-best predictions to generate."})
    max_answer_length: int = field(default=30, metadata={"help": "Maximum length of an answer that can be generated."})

    def __post_init__(self):
        if (self.dataset_name is None and self.train_file is None and 
            self.validation_file is None and self.test_file is None):
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            for file in [self.train_file, self.validation_file, self.test_file]:
                if file is not None:
                    extension = file.split(".")[-1]
                    assert extension in ["csv", "json"], f"File {file} should be csv or json."

# -------------------- 自定义评价指标函数 --------------------
import re
import string
from transformers import EvalPrediction

# ✅ BIO strict match metrics for compute_custom_metrics()
def get_entities(seq, id2label):
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(seq):
        label = id2label[tag] if isinstance(tag, int) else tag
        if label.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(tuple(chunk))
            chunk = [label[2:], i, i]
        elif label.startswith("I-") and chunk[0] == label[2:]:
            chunk[2] = i
        else:
            if chunk[2] != -1:
                chunks.append(tuple(chunk))
            chunk = [-1, -1, -1]
    if chunk[2] != -1:
        chunks.append(tuple(chunk))
    return chunks

def compute_strict_bio_metrics(predictions, references, id2label):
    TP = FP = FN = 0
    for pred_seq, true_seq in zip(predictions, references):
        pred_ents = get_entities(pred_seq, id2label)
        true_ents = get_entities(true_seq, id2label)
        TP += sum([1 for ent in pred_ents if ent in true_ents])
        FP += len([ent for ent in pred_ents if ent not in true_ents])
        FN += len([ent for ent in true_ents if ent not in pred_ents])
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "bio_strict_precision": precision,
        "bio_strict_recall": recall,
        "bio_strict_f1": f1,
        "bio_TP": TP,
        "bio_FP": FP,
        "bio_FN": FN
    }

def normalize_text(s):
    return s.lower().strip()

def tokenize(text):
    return text.split()

def make_bio_labels(context, spans, label_name="ANSWER"):
    tokens = tokenize(context)
    token_offsets = []
    start = 0
    for token in tokens:
        offset = context.find(token, start)
        token_offsets.append((offset, offset + len(token)))
        start = offset + len(token)
    labels = ["O"] * len(tokens)
    for span_start, span_end in spans:
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_end <= span_start:
                continue
            if tok_start >= span_end:
                break
            if tok_start >= span_start and tok_end <= span_end:
                labels[i] = "B-" + label_name if labels[i] == "O" else "I-" + label_name
    return labels

def convert_to_ids(seq, label2id):
    return [label2id.get(label, 0) for label in seq]

# ✅ 正确封装的闭包版本

def build_compute_custom_metrics(eval_examples):
    def compute_custom_metrics(p: EvalPrediction):
        pred_dict = {pred["id"]: normalize_text(pred["prediction_text"]) for pred in p.predictions}

        label_paths = []
        pred_paths = []

        for item in eval_examples:
            qid = item["id"]
            context = item["context"]
            context_norm = normalize_text(context)

            answer_text = normalize_text(item["answers"]["text"][0])
            answer_start = item["answers"]["answer_start"][0]
            answer_end = answer_start + len(answer_text)
            label_seq = make_bio_labels(context_norm, [(answer_start, answer_end)])

            pred_text = pred_dict.get(qid, "")
            pred_start = context_norm.find(pred_text)
            if pred_start != -1:
                pred_end = pred_start + len(pred_text)
                pred_seq = make_bio_labels(context_norm, [(pred_start, pred_end)])
            else:
                pred_seq = make_bio_labels(context_norm, [])

            label_paths.append(label_seq)
            pred_paths.append(pred_seq)

        id2label = {0: 'O', 1: 'B-ANSWER', 2: 'I-ANSWER'}
        label2id = {v: k for k, v in id2label.items()}

        label_paths = [convert_to_ids(seq, label2id) for seq in label_paths]
        pred_paths = [convert_to_ids(seq, label2id) for seq in pred_paths]

        return compute_strict_bio_metrics(pred_paths, label_paths, id2label)
    return compute_custom_metrics


# ✅ 主接口函数（支持 eval_examples）
from transformers import EvalPrediction


def normalize_text(s):
    return s.lower().strip()

def tokenize(text):
    return text.split()

def make_bio_labels(context, spans, label_name="ANSWER"):
    tokens = tokenize(context)
    token_offsets = []
    start = 0
    for token in tokens:
        offset = context.find(token, start)
        token_offsets.append((offset, offset + len(token)))
        start = offset + len(token)
    labels = ["O"] * len(tokens)
    for span_start, span_end in spans:
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_end <= span_start:
                continue
            if tok_start >= span_end:
                break
            if tok_start >= span_start and tok_end <= span_end:
                labels[i] = "B-" + label_name if labels[i] == "O" else "I-" + label_name
    return labels

def convert_to_ids(seq, label2id):
    return [label2id.get(label, 0) for label in seq]

# ✅ 主接口函数
from transformers import EvalPrediction

def compute_custom_metrics(p: EvalPrediction):
    pred_dict = {pred["id"]: normalize_text(pred["prediction_text"]) for pred in p.predictions}

    label_paths = []
    pred_paths = []

    for item in p.label_ids:
        qid = item["id"]
        context = item["context"]
        context_norm = normalize_text(context)

        answer_text = normalize_text(item["answers"]["text"][0])
        answer_start = item["answers"]["answer_start"][0]
        answer_end = answer_start + len(answer_text)
        label_seq = make_bio_labels(context_norm, [(answer_start, answer_end)])

        pred_text = pred_dict.get(qid, "")
        pred_start = context_norm.find(pred_text)
        if pred_start != -1:
            pred_end = pred_start + len(pred_text)
            pred_seq = make_bio_labels(context_norm, [(pred_start, pred_end)])
        else:
            pred_seq = make_bio_labels(context_norm, [])

        label_paths.append(label_seq)
        pred_paths.append(pred_seq)

    id2label = {0: 'O', 1: 'B-ANSWER', 2: 'I-ANSWER'}
    label2id = {v: k for k, v in id2label.items()}

    label_paths = [convert_to_ids(seq, label2id) for seq in label_paths]
    pred_paths = [convert_to_ids(seq, label2id) for seq in pred_paths]

    return compute_strict_bio_metrics(pred_paths, label_paths, id2label)




# -------------------- 主函数与数据预处理（保持与原始代码一致） --------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(f"Output directory ({training_args.output_dir}) exists and is not empty. Use --overwrite_output_dir.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    # 加载数据集（支持 Squad 格式数据）
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1] if data_args.train_file is not None else "json"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training a new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("This script only works for models with a fast tokenizer.")

    # 确定数据集列名（假定 Squad 格式）
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f"max_seq_length ({data_args.max_seq_length}) > model's max length ({tokenizer.model_max_length}).")
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # 数据预处理函数：训练集
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        if data_args.do_not_use_token_type_ids and "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples

    # 数据预处理函数：验证/预测集
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        if data_args.do_not_use_token_type_ids and "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        if data_args.version_2_with_negative:
            formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # 使用自定义评价函数 compute_custom_metrics，输出 6 个指标
    trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=build_compute_custom_metrics(eval_examples),  # ✅ 改这里
)


    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None else last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name
        trainer.push_to_hub(**kwargs)

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
