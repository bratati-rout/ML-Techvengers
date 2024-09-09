import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk

# Download necessary NLTK data
# nltk.download('punkt')

def fetch_text_from_url(url):
    """
    Fetches text from a given URL.

    Args:
        url (str): URL of the webpage to scrape.

    Returns:
        str: Raw text extracted from the webpage.
    """
    response = requests.get(url)
    article = response.text
    soup = BeautifulSoup(article, 'lxml')
    paragraphs = soup.find_all('p')
    return " ".join([p.text for p in paragraphs])

def clean_text(text):
    """
    Cleans the input text by removing references, extra spaces, and non-alphabet characters.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\[[0-9]*\]', '', text)  # Remove reference numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only alphabets
    return text

def extract_sentences_with_keywords(text, keywords):
    """
    Extracts sentences that contain specific keywords.

    Args:
        text (str): The text to search for keywords.
        keywords (set): A set of keywords to look for in the text.

    Returns:
        list: A list of sentences that contain the keywords.
    """
    sentences = sent_tokenize(text)
    keyword_sentences = [sentence for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in keywords)]
    return keyword_sentences

def extractive_summary_with_keywords(text, keywords, num_sentences=7):
    """
    Generates an extractive summary from the given text using TF-IDF and keyword boosting.

    Args:
        text (str): The text to summarize.
        keywords (set): A set of keywords to prioritize in the summary.
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: Extractive summary focusing on keywords.
    """
    # Clean and tokenize sentences
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(text)
    
    # Generate TF-IDF matrix for sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate sentence scores based on TF-IDF
    sentence_scores = tfidf_matrix.sum(axis=1).flatten().tolist()[0]
    
    # Boost scores for sentences containing keywords
    keyword_sentences = extract_sentences_with_keywords(text, keywords)
    for i, sentence in enumerate(sentences):
        if sentence in keyword_sentences:
            sentence_scores[i] *= 1.5  # Boost score by 50% if the sentence contains a keyword

    # Get top N sentences for the summary
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])

    return summary

def summarize_policy_with_keywords(text, keywords, num_sentences=7):
    """
    Generates an extractive summary from the given text focusing on specific keywords.

    Args:
        text (str): The text to summarize.
        keywords (set): A set of keywords to prioritize in the summary.
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: Extractive summary of the policy document focusing on keywords.
    """
    summary = extractive_summary_with_keywords(text, keywords, num_sentences=num_sentences)
    return summary

# Streamlit Frontend
st.title("Policy Summarizer & Keyword Extractor")

# Create two tabs for URL and text input
tab1, tab2 = st.tabs(["Summarize from URL", "Summarize from Copy-Pasted Text"])

# Predefined keywords
keywords = {'interest', 'ROI', 'principal', 'policy', 'insurance'}

with tab1:
    # Input fields for URL
    url = st.text_input("Enter the policy URL:")
    num_sentences_url = st.number_input("Number of sentences for the summary (URL):", min_value=1, max_value=100, value=7)

    if st.button("Summarize from URL"):
        if url:
            raw_text = fetch_text_from_url(url)
            summary = summarize_policy_with_keywords(raw_text, keywords, num_sentences=num_sentences_url)
            st.subheader("Extractive Summary from URL:")
            st.write(summary)
        else:
            st.write("Please provide a valid URL.")

with tab2:
    # Input fields for copy-pasted text
    text_input = st.text_area("Paste your policy text here:")
    num_sentences_text = st.number_input("Number of sentences for the summary (Text):", min_value=1, max_value=100, value=7, key='text_summary')

    if st.button("Summarize from Text"):
        if text_input:
            summary = summarize_policy_with_keywords(text_input, keywords, num_sentences=num_sentences_text)
            st.subheader("Extractive Summary from Pasted Text:")
            st.write(summary)
        else:
            st.write("Please paste some text for summarization.")
