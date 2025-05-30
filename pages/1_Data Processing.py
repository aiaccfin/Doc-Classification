import streamlit as st
from streamlit_extras.stateful_button import button

import pandas as pd
import config

from utils import streamlit_components, dataset_processing

streamlit_components.streamlit_ui('🦣 PDF Processing to pkl')

st.text(f"dataset invoice folder: {config.DS_INV}")
st.text(f"dataset receipt folder: {config.DS_RE}")
st.text(f"dataset bs folder: {config.DS_BS}")
st.text(f"pkl: {config.DS_finalframe}")

if button("Convert?", key="but5"):
    inv_text= dataset_processing.convert_files_to_text(config.DS_INV)
    st.info('invoice...')
    inv_df  = dataset_processing.text_processing(inv_text, 'Invoice', 'invoice, bills')

    re_text = dataset_processing.convert_files_to_text(config.DS_RE)
    st.info('receipt...')
    re_df = dataset_processing.text_processing(re_text, 'Receipt', 'Receipt')

    bs_text = dataset_processing.convert_files_to_text(config.DS_BS)
    st.info('bs...')
    bs_df = dataset_processing.text_processing(bs_text, 'Bank Statement', 'Bank Statements, bank document')

    frames = [inv_df, re_df, bs_df]
    finalframe = pd.concat(frames, sort=False)
    finalframe = finalframe[['Identifiers', 'Text_Data', 'Category']]
    finalframe = finalframe.reset_index(drop=True)
    finalframe.to_pickle(config.DS_finalframe)
    st.dataframe(finalframe)
    
    st.success(f"pkl saved at: {config.DS_finalframe}")