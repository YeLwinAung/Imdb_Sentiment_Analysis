import sys
import subprocess

def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("tensorflow")
install("torch")
install("transformers")
install("datasets")
install("streamlit")

import streamlit as st
import tensorflow as tf
import torch
import transformers

st.title("IMDb Sentiment Analysis Application")

st.write("Environment initialized")

st.write("TensorFlow version:", tf.__version__)
st.write("PyTorch version:", torch.__version__)
st.write("Transformers version:", transformers.__version__)