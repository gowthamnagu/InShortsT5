import pandas as pd
from datasets import Dataset
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import os
import tensorflow as tf
import evaluate


df=pd.read_csv("news_summary.csv",encoding="ISO-8859-1")

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

df=df.rename(columns={'ctext':'input','text':'target'})
df.isnull().sum()
df=df.dropna()
df.isnull().sum()

dataset=Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

#print(train_dataset['input'][0])

input=[inp for inp in train_dataset["input"]]
#print(input)

##preprocess

def preprocess(examples):
    target_text=examples['target']
    inputs = ["summarize: " + inp for inp in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_data = dataset.map(preprocess, batched=True)

first_row=tokenized_data['train'][0]
#print(f"input_ids: {first_row['input_ids']}")
#print(f"label: {first_row['labels']}")

#print(f"decode_input_ids: {tokenizer.decode(first_row['input_ids'],skip_special_tokens=True)}")
#print(f"decode_label: {tokenizer.decode(first_row['labels'],skip_special_tokens=True)}")

tf_train_dataset = tokenized_data["train"].to_tf_dataset(columns=["input_ids", "attention_mask"], label_cols="labels", shuffle=True,batch_size=2)
tf_val_dataset = tokenized_data["test"].to_tf_dataset(columns=["input_ids", "attention_mask"], label_cols="labels", shuffle=False, batch_size=2)

# Compile and fit
model.compile(optimizer="adam")
model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=1)

model.save_pretrained("t3-news-summary")
tokenizer.save_pretrained("t3-news-summary")