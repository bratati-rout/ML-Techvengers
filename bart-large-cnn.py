import streamlit as st
from transformers import pipeline

# Load a pre-trained summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def summarize_text(text, max_length=130, min_length=30, do_sample=False):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']

# Streamlit App
st.title("Text Summarization App")
st.write("Enter the text you want to summarize below:")

# Define character limit
CHARACTER_LIMIT = 4000

# Text input with character count
input_text = st.text_area(
    "Input Text", 
    height=200, 
    max_chars=CHARACTER_LIMIT, 
    help=f"Character limit: {CHARACTER_LIMIT} characters"
)

# Show the character count
st.write(f"Character count: {len(input_text)}/{CHARACTER_LIMIT}")

# Parameters
max_len = st.slider("Max Length of Summary", min_value=50, max_value=500, value=130)
min_len = st.slider("Min Length of Summary", min_value=10, max_value=100, value=30)

# Summarize button with character limit check
if st.button("Summarize"):
    if len(input_text) == 0:
        st.error("Please enter some text to summarize.")
    elif len(input_text) > CHARACTER_LIMIT:
        st.error(f"Input text exceeds the character limit of {CHARACTER_LIMIT}. Please shorten your text.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_text(input_text, max_length=max_len, min_length=min_len)
        st.subheader("Summary:")
        st.write(summary)

# Footer
st.write("---")
st.write("This is a simple text summarization app using a pre-trained BART model.")
