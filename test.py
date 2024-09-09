import spacy
import streamlit as st
from collections import Counter

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extractive_summary(text, min_percentage=0.2, max_percentage=0.5):
    # Process the text
    doc = nlp(text)

    # Extract sentences
    sentences = [sent for sent in doc.sents]

    # Create a list of words from the text (excluding stopwords, punctuation, etc.)
    words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Count word frequencies
    word_freq = Counter(words)

    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent in sentence_scores:
                    sentence_scores[sent] += word_freq[word.text.lower()]
                else:
                    sentence_scores[sent] = word_freq[word.text.lower()]

    # Determine the number of sentences for the summary
    num_sentences = len(sentences)
    min_sentences = max(int(num_sentences * min_percentage), 1)  # Ensure at least 1 sentence
    max_sentences = max(int(num_sentences * max_percentage), min_sentences)

    # Sort sentences by score and select the top sentences
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    summary = " ".join([sent.text for sent in summarized_sentences])

    return summary

def main():
    st.title("Policy Document Summarizer")

    # Upload text file
    uploaded_file = st.file_uploader("Choose a policy/legal document", type="txt")

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.write("Original Document:")
        st.text_area("Document Content", text, height=300)

        # Summary parameters
        st.write("Generating Summary...")
        summary = extractive_summary(text)
        st.write("Summary:")
        st.text_area("Summarized Content", summary, height=300)

if __name__ == "__main__":
    main()
