
# 📰 News Summarizer

**News Summarizer** is a deep learning-based application built using a fine-tuned **T5-small** model with **TensorFlow** for *abstractive* news summarization. The goal is to automatically generate concise and meaningful summaries from longer news articles.

Unlike extractive approaches that merely copy sentences from the original text, this summarizer generates entirely new summaries, mimicking human-like abstraction and paraphrasing using the **T5 architecture**.

## 📌 Dataset

The model was fine-tuned on a Kaggle dataset of **4,500 news records**, where each sample includes:

- `ctext`: the full news article
- `text`: the medium-length summary

## 📽️ Demo

![text_summarizer_demo](https://github.com/gowthamnagu/news-summarization/blob/main/demo/demo.gif)

## 🧠 Project Overview
This project leverages a **fine-tuned T5-small model** using TensorFlow for summarizing news content. It transforms long-form text into short, informative summaries that preserve the core meaning.

Key Features:
- Compare **fine-tuned** vs **pretrained** T5 summarization
- Adjustable generation parameters (beam search, max length, etc.)
- ROUGE evaluation metrics integrated into the app
- Fully interactive web app built with **Streamlit**

## 🛠️ Tech Stack
- **TensorFlow** – For model training and inference (fine-tuned T5).
- **Transformers** (Hugging Face) – For leveraging the T5-small model and tokenizer.
- **Evaluate** – For computing ROUGE metrics during summarization evaluation.
- **Pandas / Datasets** – For handling and preprocessing text data.
- **Streamlit** – To build the interactive web interface for summarizing text using pretrained and fine-tuned models.
- **Git & GitHub** – For version control and open-source hosting.
- **Git LFS (Large File Storage)** – For storing and managing large model weights (tf_model.h5).
## 🚀 Installation
    

- ### 1. Clone the repository
    
       git clone  https://github.com/gowthamnagu/news-summarization.gi

-  ### 2. Set up a virtual environment

        conda create -p venv python=3.11 -y
        conda activate venv/

-  ### 3. Install dependencies 

        pip install -r requirements.txt

     
  ### 4. Running the Application

        Run the main file to start the application:

        streamlit run app.py.

## Project Structure    
```    
└── news_summarizer/
    ├── t5-ns-12/
    │   ├── tf_model.h5      #finetunied model file
    │   └── tokenizer.json
    ├── app.py              #streamlit applicaiton for new_summarization
    ├── dockerfile         #to containerize news_summary application 
    ├── finetune_t5.py      
    ├── ts_finetuned.py     #contains functionality for generating summary based on finetune model(t5-ns-12)
    ├── ts_pretuned.py      #contains functionality for generating summary based on pretuned model (t5-small)
    ├── news_summary.csv    #dataset file for finetuning the t5-small file
    └── requirements.txt    #Lists all the dependencies required for the project.
```
