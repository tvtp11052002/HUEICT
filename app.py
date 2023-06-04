import streamlit as st

st.set_page_config(page_title="Predict Result of WORLDCUP 2026✨", page_icon="⚽",
                   layout="wide", initial_sidebar_state='expanded')
import io
import requests
import json
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
# download the stopwords from NLTK
import nltk


@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')

load_nltk()

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import TextVectorization
from sentence_transformers import SentenceTransformer

from utils.image_func import *
from utils.nlp_func import *
from utils.process import *

st.text_input("Đây là Phong")
