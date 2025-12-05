import torch
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from PIL import Image,ImageDraw, ImageFont
from datasets import load_dataset
import pandas as pd

import evaluate
from transformers import BrosProcessor, BrosForTokenClassification, BrosSpadeEEForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator

def get_data():
    global label2id 
    global id2label 
    global label_list
    funsd = load_dataset("nielsr/funsd", trust_remote_code=True)
    label_list = funsd["train"].features["ner_tags"].feature.names  
    id2label = {v:k for v,k in enumerate(label_list)}
    label2id = {k:v for v,k in enumerate(label_list)}
    return funsd, id2label, label2id, label_list


def get_model(id2label, label2id, label_list):
    global processor
    global model
    processor = BrosProcessor.from_pretrained("naver-clova-ocr/bros-base-uncased")
    model = BrosForTokenClassification.from_pretrained("naver-clova-ocr/bros-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)
    return processor, model


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def tokenize_words(batch, processor):
  encodings = processor.tokenizer(
    batch["words"],
    is_split_into_words=True,
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt"
  )

  batch_normalized_bboxes, encoded_labels = [], []
  for idx, (bboxes, img, labels) in enumerate(zip(batch["bboxes"], batch["image"], batch["ner_tags"])):
    width, height = img.size
    normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in bboxes]

    # Align boxes to sub words
    aligned_boxes, aligned_labels = [], []
    for word_id in encodings.word_ids(batch_index=idx):
      if word_id is None:
        aligned_boxes.append([0, 0, 0, 0])
        aligned_labels.append(-100)
      else:
        aligned_boxes.append(normalized_bboxes[word_id])
        aligned_labels.append(labels[word_id])

    batch_normalized_bboxes.append(aligned_boxes)
    encoded_labels.append(aligned_labels)

  encodings['bbox'] = batch_normalized_bboxes
  encodings['labels'] = encoded_labels

  return encodings


def preprocess(data):
    from functools import partial
    train_dataset = data["train"].map(partial(tokenize_words, processor=processor), batched=True, remove_columns=data["train"].column_names)
    val_dataset = data["test"].map(partial(tokenize_words, processor=processor), batched=True, remove_columns=data["train"].column_names)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    return train_dataset, val_dataset


def get_training_args(args):
    return TrainingArguments(
    output_dir="./bros-funsd-finetuned",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=args.epochs,
    save_strategy="best",
    save_total_limit=2,
    optim="adamw_torch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to = 'none',
    metric_for_best_model="eval_f1",
    )


def compute_metrics(p):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=-1)

  true_preds = [
      [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  true_labels = [
      [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  results = metric.compute(predictions=true_preds, references=true_labels)

  return {
      "precision": results["overall_precision"],
      "recall": results["overall_recall"],
      "f1": results["overall_f1"],
      "accuracy": results["overall_accuracy"],
  }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    funsd, id2label, label2id, label_list = get_data()
    print("[bros funsd] Got data")
    processor, model = get_model(id2label,label2id, label_list)
    print("[bros funsd] Got model")
    train, val = preprocess(funsd)
    print("[bros funsd] Preprocessed")
    training_args = get_training_args(args)

    global metric
    metric = evaluate.load("seqeval")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DefaultDataCollator(),
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    model = trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="amount of epochs", default=10, type=int)
    args= parser.parse_args()
    main(args)