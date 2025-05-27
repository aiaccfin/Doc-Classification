import streamlit as st, os


def streamlit_ui(main_title):
    st.set_page_config(page_title='AI Auto Accounting 👋', page_icon="💯", ),
    st.title(main_title)  # not accepting default

    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://omnidevx.netlify.app/logo/aiacc.png);
                background-size: 180px; /* Set the width and height of the image */
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 60px 1px;
            }
        </style>
        """,
                unsafe_allow_html=True,
                )


def general():
    st.markdown('''
        Extracting attachment from email, we need classify them.
    '''
                )
