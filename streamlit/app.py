import streamlit as st

import pandas as pd
from image_recommend import *


from object_detection import *

app_formal_name = "Image Caption"
# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_formal_name,
)


st.title("Image Recommendation")

st.text('')
st.text('')



st.markdown('#### Search image using text (Enter Text) :')
default_value_goes_here="person walking with dog "
user_input = st.text_input(" ", default_value_goes_here)
if st.button('Search',key="text"):
    img1,img2=predict_image(user_input)
    col1, col2 = st.beta_columns([1, 2])
    with col1:
        st.image(img1)
    with col2:
        st.image(img2)


st.text('')
st.text('')

st.markdown('#### Object detection (Upload Image) :')

uploaded_file = st.file_uploader("Choose an image...")
if st.button('Search',key="upload"):
    if uploaded_file is not None:
        col1, col2 = st.beta_columns([1, 2])
        image = Image.open(uploaded_file)
        with col1:
            st.image(image)
        detection_image,frequencies=object_detections(image)
        print(frequencies)
        with col2:
            st.image(detection_image)

        st.markdown("#### Tags")
        st.text(frequencies)


