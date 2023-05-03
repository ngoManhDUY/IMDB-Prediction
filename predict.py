import streamlit as st
import pickle
import numpy as np
from streamlit_ace import *
import base64

from PIL import Image

def load_model():
    with open("Save_model.pkl", 'rb' ) as file:
        data = pickle.load(file)
    return data

data = load_model() 

ran_for = data["model1"]
knn = data["model2"]

id_Genre = data["id_Genre"]

def predict_page():
    st.title("IMDB rating and Movie quality prediction: ")

    st.write("""### We need some information of the Movie""")

    # Set background image

    Gerne = ("Action", "Adventure", "Horror" , "Animation", "Comedy", "Biography", "Drama", "Crime", 
                  "Mystery", "Thriller")
    
    Gerne = st.selectbox("Gernes: ",Gerne)
    
    Runtime = st.text_input('Enter Runtime(Minutes) of the Movie', value='0')
    try:
        number = float(Runtime)
        st.write(f'Runtime: {number}')
    except ValueError:
        st.write('Please enter right number !')

    vote = st.text_input('Enter Number of votes by IMDB users:', value='0')
    try:
        number = int(vote)
        st.write(f'Total vote: {number}')
    except ValueError:
        st.write('Please enter right number !')

    Revenue = st.text_input('Enter Revenue (Millions Dollar) of the movie:', value='0')
    try:
        number = int(Revenue)
        st.write(f'Revenue : {number} Million Dollar')
    except ValueError:
        st.write('Please enter right number !')

    meta = st.text_input('Enter Metascore of the movie:', value='0')
    try:
        number = int(meta)
        st.write(f'Movie Metascore: {number}')
    except ValueError:
        st.write('Please enter right number !')

    imdb = st.text_input('Enter the actual IMDB rating of the movie:', value='0')
    try:
        number = int(meta)
        st.write(f'The real IMDB rating: {imdb}')
    except ValueError:
        st.write('Please enter right number !')


    ok = st.button("My Model prediction: ")
    if ok:
        
        z = np.array([[Gerne, Runtime ,vote ,Revenue,meta]])
        z[ :,0] = id_Genre.transform(z[ :,0])
        z = z.astype(float)

        l = np.array([[Gerne, Runtime ,vote ,Revenue,meta,imdb]])
        l[ :,0] = id_Genre.transform(l[ :,0])
        l = l.astype(float)
        

        imdb = ran_for.predict(z)
        st.subheader(f"The estimated IMDB rated is {imdb[0]:.2f}")

        quality = knn.predict(l)
        st.subheader(f"And the prediction of the movie quality is: {quality[0]}")
