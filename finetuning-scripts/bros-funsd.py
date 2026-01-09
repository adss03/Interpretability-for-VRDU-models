'''
results:
{
  'eval_loss': 1.6927354335784912, 
  'eval_precision': 0.5546666666666666, 
  'eval_recall': 0.611964694344557, 
  'eval_f1': 0.5819086105066832, 
  'eval_accuracy': 0.6861001051184604, 
  'epoch': 100.0
  'train_loss': 0.3588285522460937, 
  'epoch': 100.0
}         
'''

import torch
import itertools
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from PIL import Image,ImageDraw, ImageFont
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

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


def get_model(args, id2label, label2id, label_list):
    global processor
    global model
    processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")
    
    if args.spade == True:
      model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)
      print('[bros funsd] SpadeEE loaded')
    else:
      model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)
      print('[bros funsd] Bros Model loaded')
    return processor, model


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
 

def tokenize_words(batch, args, processor):
  encodings = processor.tokenizer(
    batch["words"],
    is_split_into_words=True,
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
    return_token_type_ids=True
  )

  def make_box_first_token_mask(bboxes, input_ids, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    # get the length of each box
    tokens_length_list: list[int] = [len(l) for l in input_ids]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    # filter out the indices that are out of max_seq_length
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    # set box_start_token_indices to True
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask
  

  batch_normalized_bboxes, encoded_labels = [], []
  for idx, (words, bboxes, img, labels) in enumerate(zip(batch["words"], batch["bboxes"], batch["image"], batch["ner_tags"])):
    width, height = img.size
    normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in bboxes]

    # Align boxes to sub words
    aligned_boxes, aligned_labels = [], []
    for word_id in encodings.word_ids(batch_index=idx):
      if word_id in [None,0,100,101,102,103]:
      if word_id in [None,0,100,101,102,103]:
        aligned_boxes.append([0, 0, 0, 0])
        aligned_labels.append(-100)
      else:
        aligned_boxes.append(normalized_bboxes[word_id])
        aligned_labels.append(labels[word_id])
    
    # box_first_token_mask = make_box_first_token_mask()

    batch_normalized_bboxes.append(aligned_boxes)
    encoded_labels.append(aligned_labels)

  encodings['bbox'] = batch_normalized_bboxes

  # if args.spade:
  #   encodings["bbox_first_token_mask"] = [make_box_first_token_mask(boxes, ids) for boxes, ids in zip(encodings["bbox"], encodings["input_ids"])]

  encodings['labels'] = encoded_labels
  print(encodings.keys())
  return encodings


def preprocess(data, args):
    from functools import partial
    train_dataset = data["train"].map(partial(tokenize_words, args=args, processor=processor), batched=True, remove_columns=data["train"].column_names)
    val_dataset = data["test"].map(partial(tokenize_words, args=args, processor=processor), batched=True, remove_columns=data["train"].column_names)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    return train_dataset, val_dataset


def get_training_args(args):
    # Take from config/bros in bros githu
    return TrainingArguments(
      output_dir="./bros-funsd-finetuned",
      eval_strategy="epoch",
      num_train_epochs=args.epochs,
      fp16=True,
      fp16=True,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=8,
      optim="adamw_torch",
      learning_rate=5e-5,
      lr_scheduler_type="linear",
      lr_scheduler_type="linear",
      load_best_model_at_end=True,
      push_to_hub=False,
      report_to = 'none',
      metric_for_best_model="eval_f1",
      save_strategy="best",
      save_total_limit=1
      save_strategy="best",
      save_total_limit=1
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
  
  # confusion matrix with p 

  with open("confusion_matrix.txt", 'w') as f:
    f.write(f'{[len(p) for p in predictions]}\n')
    f.write(f'{[len(l) for l in labels]}\n')
    f.write(f'{confusion_matrix([l for lab in true_labels for l in lab], [p for preds in true_preds for p in preds])}')
  
  # confusion matrix with p 

  with open("confusion_matrix.txt", 'w') as f:
    f.write(f'{[len(p) for p in predictions]}\n')
    f.write(f'{[len(l) for l in labels]}\n')
    f.write(f'{confusion_matrix([l for lab in true_labels for l in lab], [p for preds in true_preds for p in preds])}')
  return {
      "precision": results["overall_precision"],
      "recall": results["overall_recall"],
      "f1": results["overall_f1"],
      "accuracy": results["overall_accuracy"],
  }

def spade_test():
  processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

  model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

  encoding = processor.tokenizer("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
  bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
  encoding["bbox"] = bbox
  print(encoding.keys())
  outputs = model(**encoding)
  return outputs

def main(args):
    if not args.warnings:
      import warnings
      warnings.filterwarnings('ignore') 
    if args.spadetest:
      print(spade_test())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    funsd, id2label, label2id, label_list = get_data()
    print("[bros funsd] Got data")
    processor, model = get_model(args, id2label,label2id, label_list)
    print("[bros funsd] Got model")
    train, val = preprocess(funsd, args)
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
    parser.add_argument("--warnings", help="warnings?", action=argparse.BooleanOptionalAction)
    parser.add_argument("--spade", help="use the spade head", action=argparse.BooleanOptionalAction)
    parser.add_argument("--spadetest", action=argparse.BooleanOptionalAction)
    args= parser.parse_args()
    main(args)