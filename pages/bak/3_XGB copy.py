import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, image_processing

import config
import pandas as pd
import matplotlib.pyplot as plt

import pickle, requests, json

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

streamlit_components.streamlit_ui('🦣 XGBoost Classifier')

model_rf   = config.MODEL_rf
model_nbc  = config.MODEL_nbc
model_xgb  = config.MODEL_xgb
dataset_h5  = config.DS_finalframe_h5

X_train = pd.read_hdf(dataset_h5 , key='X_train')
X_test  = pd.read_hdf(dataset_h5, key='X_test')
y_train = pd.read_hdf(dataset_h5, key='y_train')
y_test  = pd.read_hdf(dataset_h5, key='y_test')


my_tags = ['Invoices', 'Receipts','Bank Statements']


if button("Continue ?", key="button4"):

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %s" % (accuracy))
    print(classification_report(y_test, y_pred, target_names=my_tags))

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('Confusion matrix:\n', conf_mat)

    labels = ['Invoices', 'Receipts','Bank Statements']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()
