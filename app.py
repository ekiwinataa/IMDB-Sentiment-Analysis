import streamlit as st
import pickle
from utils import preprocessing_text

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon=":pig:"
)

if 'model' not in st.session_state:
    model = pickle.load(open('model.sav', 'rb'))
    st.session_state['model'] = model

st.title('IMDB Sentiment Analysis')

review = st.text_area('Reviewed text to analyze', 
                     "I don't know how or why this film has a meager...")

if st.button('Analyze'):
    review_prep = preprocessing_text(review)
    result = st.session_state['model'].predict([review_prep])
    if result.tolist()[0] == 0:
        result = 'Positive Review'
    else:
        result = 'Negative Review'
    st.write('Sentiment:', result)
else:
    st.write('waiting')