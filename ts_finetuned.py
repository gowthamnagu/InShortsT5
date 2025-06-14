import evaluate
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf
import os
import streamlit as st

@st.cache_resource
def load_finetuned_model():
    model_name = "t5-ns-12"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer

model, tokenizer = load_finetuned_model()

rouge = evaluate.load('rouge')
def summarize_text_finetuned(input,max_length,min_length,num_beams,length_penalty,early_stopping):
    input_text = "summarize: " + input
    inputs = tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=early_stopping
    )
    decode_text=tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    evaluate_result = rouge.compute(predictions=[decode_text], references=[input])

    return decode_text, evaluate_result


#rouge = evaluate.load('rouge')
#results = rouge.compute(predictions=[summary], references=[article])
