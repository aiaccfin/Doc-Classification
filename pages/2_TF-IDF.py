import streamlit as st, pickle
from streamlit_extras.stateful_button import button

import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import config

from utils import streamlit_components
streamlit_components.streamlit_ui('🦣 Term Frequency-Inverse Doc Frequency')


st.text(f"pkl: {config.DS_finalframe}")
st.text(f"pkl: {config.DS_finalframe_h5}")


finalframe = pd.read_pickle(config.DS_finalframe)


if button("TF-IDF?", key="button1"):
    # Converting the text data into vectors using TF-IDF
    # Generating 1000 features for the input for the model
    tfidfconverter = TfidfVectorizer(max_features=2000, stop_words=stopwords.words('english'))
    X = pd.DataFrame(tfidfconverter.fit_transform(finalframe['Text_Data']).toarray())
    st.dataframe(X)
    # X.columns = range(X.shape[1])
    labelencoder = LabelEncoder()  # Converting the labels to numeric labels
    y = labelencoder.fit_transform(finalframe['Category'])


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Save all datasets in one HDF5 file
    X_train.to_hdf(config.DS_finalframe_h5, key='X_train', mode='w')
    X_test.to_hdf(config.DS_finalframe_h5, key='X_test', mode='a')
    pd.Series(y_train).to_hdf(config.DS_finalframe_h5, key='y_train', mode='a')
    pd.Series(y_test).to_hdf(config.DS_finalframe_h5, key='y_test', mode='a')

    with open(config.DS_vector, 'wb') as f:
            pickle.dump(tfidfconverter, f)

    st.success(f"✅ TF-IDF vectorizer saved to: {config.DS_vector}")
    st.success(f"✅ Training data saved to: {config.DS_finalframe_h5}")
