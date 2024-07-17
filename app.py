import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

# Function to plot CI plot
def plot_ci(variable, outcome, sample, data):
    # Extract the relevant row for the selected variable, outcome, and sample
    variable_data = data[(data['Variable'] == variable) & (data['Outcome'] == outcome) & (data['Sample'] == sample)]

    # Extract estimates and confidence intervals
    groups = ['G1', 'G2', 'G3', 'G4', 'G5']
    estimates = variable_data[[f'{g}_Estimate' for g in groups]].values.flatten()
    ci_lowers = variable_data[[f'{g}_CI_lower' for g in groups]].values.flatten()
    ci_uppers = variable_data[[f'{g}_CI_upper' for g in groups]].values.flatten()

    # Create the plot using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(groups, estimates, marker='o', linestyle='-', color='blue', label='Estimate')

    # Plot the confidence intervals
    ax.fill_between(groups, ci_lowers, ci_uppers, color='blue', alpha=0.2, label='95% CI')

    ax.set_title(f'CI Plot for {variable} - {outcome} - {sample}')
    ax.set_xlabel('Groups')
    ax.set_ylabel('Estimates')
    ax.legend()

    st.pyplot(fig)

# Streamlit app
st.set_page_config(page_title='CI Plot Viewer', page_icon="🧡", layout="wide")

# File uploader
uploaded_file = "data.xlsx"

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    covariates = data['Variable'].unique()
    outcomes = data['Outcome'].unique()
    samples = data['Sample'].unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Plot1")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            selected_covariate1 = st.selectbox("Covariate (Plot 1)", covariates, key='cov1')
        with scol2:
            selected_outcome1 = st.selectbox("Outcome (Plot 1)", outcomes, key='out1')
        with scol3:
            selected_sample1 = st.selectbox("Sample (Plot 1)", samples, key='sam1')
        plot_ci(selected_covariate1, selected_outcome1, selected_sample1, data)
        
        st.subheader("Plot3")

        scol4, scol5, scol6 = st.columns(3)
        with scol4:
            selected_covariate2 = st.selectbox("Covariate (Plot 2)", covariates, key='cov2')
        with scol5:
            selected_outcome2 = st.selectbox("Outcome (Plot 2)", outcomes, key='out2')
        with scol6:
            selected_sample2 = st.selectbox("Sample (Plot 2)", samples, key='sam2')
        plot_ci(selected_covariate2, selected_outcome2, selected_sample2, data)
        
    with col2:
        st.subheader("Plot2")

        scol7, scol8, scol9 = st.columns(3)
        with scol7:
            selected_covariate3 = st.selectbox("Covariate (Plot 3)", covariates, key='cov3')
        with scol8:
            selected_outcome3 = st.selectbox("Outcome (Plot 3)", outcomes, key='out3')
        with scol9:
            selected_sample3 = st.selectbox("Sample (Plot 3)", samples, key='sam3')
        plot_ci(selected_covariate3, selected_outcome3, selected_sample3, data)
        
    with col2:
        st.subheader("Plot4")

        scol10, scol11, scol12 = st.columns(3)
        with scol10:
            selected_covariate4 = st.selectbox("Covariate (Plot 4)", covariates, key='cov4')
        with scol11:
            selected_outcome4 = st.selectbox("Outcome (Plot 4)", outcomes, key='out4')
        with scol12:
            selected_sample4 = st.selectbox("Sample (Plot 4)", samples, key='sam4')
        plot_ci(selected_covariate4, selected_outcome4, selected_sample4, data)

st.data_editor(data)