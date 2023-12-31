import streamlit as st
import pickle
from utils import preprocessing_text

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon=":movie_camera:"
)

if 'model' not in st.session_state:
    model = pickle.load(open('model.sav', 'rb'))
    st.session_state['model'] = model

st.title('IMDB Sentiment Analysis')

review = st.text_area('Reviewed text to analyze', 
                     "This movie makes Peter an elf in Robin Hood costume instead of a human boy in probably-not-Robin-Hood-costume and ignores all the persona features in him that really matter. This movie makes Wendy a babbling idiot. And poor Captain Hook a TOTAL clown. And of course as every Disney cartoon must have a character which has had too many hits in the head, they made one of the Lost Boys that one. The only character that has not been disgraced in this film is Tink. The only star is for her.<br /><br />The story itself then? The Darling parents don't even get the time to notice their kids are gone!!! Probably one of the most significant point in the original story and they ruined it! Also the famous nursery scene between Peter Pan and Wendy is a stunning piece of- There are no thimbles and no acorns - one of the little things that makes the original story such a unique one. It's a wonder he even had lost his shadow and she helped him stick it. (Even though to his shoes and it makes no sense to me.)<br /><br />Ruining a great story like this just to amuse children should be illegal. So know now if you haven't known it before - this Disney version does not have anything significant in common with the original story - which is not really a children's story but just a great, great story.<br /><br />This just annoys me to no end.")

if st.button('Analyze'):
    review_prep = preprocessing_text(review)
    result = st.session_state['model'].predict([review_prep])
    if result.tolist()[0] == 0:
        result = 'Positive Review'
    else:
        result = 'Negative Review'
    st.write('Sentiment:', result)
else:
    st.write('Click to Analyze')
