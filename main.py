import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model("models/simple_rnn_imdb.keras")

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.title('🎬 IMDB Movie Review Sentiment Analysis')

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader('📝 Enter Your Review')
    user_input = st.text_area(
        'Movie Review', 
        placeholder="Type your movie review here...",
        height=200,
        help="Enter any movie review text and the AI will analyze its sentiment"
    )
    
    analyze_button = st.button('🔍 Analyze Sentiment', type="primary", use_container_width=True)
    
    if not analyze_button:
        st.info('👆 Enter a movie review and click "Analyze Sentiment" to get started!')
        
        with st.expander('💡 See Example Reviews'):
            st.write('**Positive Example:**')
            st.write('"This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."')
            
            st.write('**Negative Example:**')
            st.write('"Terrible movie with poor acting and a confusing storyline. Complete waste of time."')

with col2:
    st.subheader('📊 Analysis Results')
    
    if analyze_button:
        if user_input.strip():
            with st.spinner('Analyzing sentiment...'):
                preprocessed_input = preprocess_text(user_input)
                prediction = model.predict(preprocessed_input, verbose=0)
            
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            sentiment_emoji = '😊' if sentiment == 'Positive' else '😞'
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
            
            st.metric(
                label="Sentiment",
                value=f"{sentiment_emoji} {sentiment}"
            )
            
            st.metric(
                label="Confidence",
                value=f"{confidence:.1%}"
            )
            
            st.subheader('📈 Prediction Score')
            score = float(prediction[0][0])
            st.progress(score)
            st.caption(f'Raw prediction score: {score:.4f} (0.5+ = Positive, <0.5 = Negative)')
            
        else:
            st.warning('⚠️ Please enter a movie review to analyze.')
    else:
        st.write('Results will appear here after analysis.')
