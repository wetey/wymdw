import streamlit as st
import pandas as pd

# TODO add pre computed examples
st.set_page_config(page_title="What your model got wrong")

st.write("# What your model got wrong")

st.markdown(
    """
    A typical NLP classification task work flow is fine-tuning a language model on a specific dataset. These models rarely, if ever, achieve 100% accuracy. If you've ever wondered, hmmm where does my model fail? Well, you've come to the right place! Throught a series of visuals, you'll hopefully have a better understanding of where your model needs improving!


    We have a [DistilBERT model](https://huggingface.co/distilbert/distilbert-base-uncased) fine-tuned on the [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) from HuggingFace. Our model has 92.9% accuracy, precision, recall, and f1-score. We are concerened with that 7.1%!
    """
)


# uploaded_file = st.file_uploader(label = 'choose a file', type = 'json')

# if uploaded_file is not None:
#     bytes_data = uploaded_file.read()
