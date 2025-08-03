import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import joblib

model = joblib.load('xyz')

st.title("Image Classifier WITH DEEP LEARNING")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)


    img = img.resize((32, 32))  
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    prediction = model.predict(img_array)
    pre = np.argmax(prediction, axis=1)[0]
try:
 if pre == 0:
    st.success("UPLOADED IMAGE  Airplane")
 elif pre == 1:
    st.success("UPLOADED IMAGE: Automobile")
 elif pre == 2:
    st.success("UPLOADED IMAGE: Bird")
 elif pre== 3:
    st.success("UPLOADED IMAGECat")
 elif pre== 4:
    st.success("UPLOADED IMAGE: Deer")
 elif pre == 5:
    st.success("UPLOADED IMAGE: Dog")
 elif pre == 6:
    st.success("UPLOADED IMAGE: : Frog")
 elif pre== 7:
    st.success("UPLOADED IMAGE: : Horse")
 elif pre== 8:
    st.success("UPLOADED IMAGE:  Ship")
 elif pre== 9:
    st.success("UPLOADED IMAGE:  Truck")
 else:
    st.write("Unknown Class")
except:
    st.error("Error in prediction")

