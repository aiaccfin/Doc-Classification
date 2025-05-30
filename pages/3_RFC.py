import streamlit as st
from streamlit_extras.stateful_button import button
from utils import dataset_processing, streamlit_components, image_processing

import os, config
import pandas as pd
import matplotlib.pyplot as plt

import pickle, requests, json

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

streamlit_components.streamlit_ui('🦣 Random Forest Classifier')

model_rf   = config.MODEL_rf

dataset_h5  = config.DS_finalframe_h5

X_train = pd.read_hdf(dataset_h5 , key='X_train')
X_test  = pd.read_hdf(dataset_h5, key='X_test')
y_train = pd.read_hdf(dataset_h5, key='y_train')
y_test  = pd.read_hdf(dataset_h5, key='y_test')

my_tags = ['Invoices', 'Receipts','Bank Statements']

if button("Run?", key="button2"):

    classifier = RandomForestClassifier(n_estimators=1200, random_state=1)  # defining 1000 nodes
    rf = classifier.fit(X_train, y_train)

    pickle.dump(rf, open(model_rf, 'wb'))

    y_pred = classifier.predict(X_test)


    st.text('Accuracy: %s' % accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred, target_names=my_tags))

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    st.text('Confusion matrix:')
    st.text(conf_mat)

    
    # labels = ['Invoices', 'Receipts','Bank Statements']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + my_tags)
    ax.set_yticklabels([''] + my_tags)

    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    st.pyplot(fig)