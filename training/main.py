from transformers import TrainingArguments, Trainer
import sys
import numpy as np
from torch import cuda
import os
import numpy as np
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
from torch import cuda
from datasets import load_dataset

cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def get_model_and_tokenizer(model_name = 'distilbert-base-uncased', num_labels = 2):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels).to(device)

    return model, tokenizer

def compute_metrics(eval_pred):

    accuracy_metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    loss = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    precision = precision_metric.compute(predictions=predictions, references=labels,average = 'weighted')['precision']
    recall = recall_metric.compute(predictions=predictions, references=labels,average = 'weighted')['recall']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average = 'weighted')['f1']

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'loss':loss}

def tokenize_function(examples):
    device = 'cuda' if cuda.is_available() else 'cpu'
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length = 512, return_tensors = 'pt').to(device)

#get model and tokenizer from HF
print("get model and tokenizer")
model, tokenizer = get_model_and_tokenizer()

dataset = load_dataset("stanfordnlp/imdb")
train_set = dataset['train']
test_set = dataset['test']

print("tokenize dataset")
#batch tokenize train and test sets
tokenized_train_dataset = train_set.map(tokenize_function, batched=True)
tokenized_test_dataset = test_set.map(tokenize_function, batched = True)

training_args = TrainingArguments(output_dir = '../../wymgw_training',
                                  evaluation_strategy = 'epoch',
                                  logging_steps = 1,
                                  num_train_epochs = 3,
                                  learning_rate = 1e-5,
                                  eval_accumulation_steps = 2)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train_dataset,
    eval_dataset = tokenized_test_dataset,
    compute_metrics = compute_metrics,
)

results = trainer.train()
model.save_pretrained('../../wymgw_training')
tokenizer.save_pretrained('../../wymgw_training')

#get predictions
predictions, labels, metrics = trainer.predict(tokenized_test_dataset)

#save metrics
trainer.log_metrics('predict', metrics)
trainer.save_metrics('predict', metrics)

predictions = np.argmax(predictions, axis=1)
predictions_df = pd.DataFrame()
predictions_df['predicted'] = predictions
predictions_df['label'] = labels
predictions_df['text'] = test_set['text']
predictions_df.to_csv('predictions.csv', header = True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''