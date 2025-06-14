import streamlit as st
st.set_page_config(page_title="Text Summarize", layout="centered")
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf
import pandas as pd
import evaluate
from ts_finetuned import summarize_text_finetuned
from ts_pretrained import summarize_text_pretrained

st.title("🌍 Text Summarize")

st.sidebar.title("🛠️ Summarization Parameters")
max_length = st.sidebar.slider("Max Length", min_value=50, max_value=300, value=150)
min_length = st.sidebar.slider("Min Length", min_value=10, max_value=150, value=80)
num_beams = st.sidebar.slider("Number of Beams", min_value=1, max_value=10, value=4)
length_penalty = st.sidebar.slider("Length Penalty", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
early_stopping = st.sidebar.checkbox("Early Stopping", value=True)

user_input = st.text_area("✏️ Enter text to summarize", height=250)

def format_rouge_score(score):
    try:
        return f"{float(score):.4f}"
    except (ValueError, TypeError):
        return "N/A"

def display_rouge_scores(evaluate_finetune, evaluate_pretrain):
    st.markdown("### 📊 ROUGE Score Comparison")

    data = {
        "ROUGE Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "Fine-tuned T5": [
            format_rouge_score(evaluate_finetune.get("rouge1")),
            format_rouge_score(evaluate_finetune.get("rouge2")),
            format_rouge_score(evaluate_finetune.get("rougeL"))
        ],
        "Pretrained T5": [
            format_rouge_score(evaluate_pretrain.get("rouge1")),
            format_rouge_score(evaluate_pretrain.get("rouge2")),
            format_rouge_score(evaluate_pretrain.get("rougeL"))
        ]
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
if st.button("Summarize"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("📝 Summarizing..."):
            summary_finetune,evaluate_finetune = summarize_text_finetuned(
                user_input,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping
            )

            summary_pretrain,evaluate_pretrain = summarize_text_pretrained(
                user_input,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping
            )
            st.markdown("### 📄 Summary:")
            st.success(summary_finetune)
            display_rouge_scores(evaluate_finetune, evaluate_pretrain)