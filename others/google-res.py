import streamlit as st
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# Load the BigBird model and tokenizer
@st.cache_resource
def load_model():
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
    return model, tokenizer

model, tokenizer = load_model()

def summarize_text(text, max_length=256, min_length=50):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=4096, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App
st.title("Text Summarization App with BigBird")
st.write("Enter the text you want to summarize below:")

# Define character limit
CHARACTER_LIMIT = 5000

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
max_len = st.slider("Max Length of Summary", min_value=500, max_value=4000, value=256)
min_len = st.slider("Min Length of Summary", min_value=500, max_value=1000, value=100)

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
st.write("This is a simple text summarization app using the BigBird model from Google Research.")
