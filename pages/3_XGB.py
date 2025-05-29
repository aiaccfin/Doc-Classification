import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, image_processing

import config
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📌 UI Title
streamlit_components.streamlit_ui('🦣 XGBoost Classifier')

# 🔄 Load models and data
model_rf   = config.MODEL_rf
model_nbc  = config.MODEL_nbc
model_xgb  = config.MODEL_xgb
dataset_h5  = config.DS_finalframe_h5

X_train = pd.read_hdf(dataset_h5 , key='X_train')
X_test  = pd.read_hdf(dataset_h5, key='X_test')
y_train = pd.read_hdf(dataset_h5, key='y_train')
y_test  = pd.read_hdf(dataset_h5, key='y_test')

my_tags = ['Invoices', 'Receipts', 'Bank Statements']

# 🚀 Trigger evaluation
if button("Continue ?", key="button4"):
    st.info("Training XGBoost model...")

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    with open(model_xgb, "wb") as f:
        pickle.dump(xgb, f)
        st.info(f"🧠 Model saved to: {model_xgb}")

    y_pred = xgb.predict(X_test)
    y_pred = xgb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=my_tags)
    conf_mat = confusion_matrix(y_test, y_pred)

    # ✅ Display results
    st.success(f"✅ Accuracy: **{accuracy:.2%}**")

    st.subheader("📊 Classification Report")
    st.text(class_report)

    st.subheader("🔄 Confusion Matrix")
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(range(len(my_tags)))
    ax.set_yticks(range(len(my_tags)))
    ax.set_xticklabels(my_tags)
    ax.set_yticklabels(my_tags)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
