import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input




# Global variables
flag_pattern = u"\U0001F1E0-\U0001F1FF"


title = "Image Classification for Predict Animal "
app_dsc = "Your App to predict the image of <i>Dog</i>, <i>Cat</i>, <i>Panda</i>"
Github_repo_info = "© By Aditya Aprianto | Source Code on Github."
Github_repo = "https://github.com/aditbest5/image-classification-fsb"


# Description Section 
st.markdown("<h1 style='text-align: center;'>"+title+"</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>"+app_dsc+"</h3>", unsafe_allow_html=True)
st.markdown("<a href= "+Github_repo+"><p style= 'text-align: center;'>"+Github_repo_info+"</p></a>", unsafe_allow_html=True)


# Load and display the logo
image = Image.open('assets/FST_Bangalore.png')
st.image(image)

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model("models/my_model.hdf5")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
  

### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'cats',
            1: 'dogs',
            2: 'panda',
}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(150,150))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))