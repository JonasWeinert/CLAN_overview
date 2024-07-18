import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

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
    
    # Extract p-values for 5-1, 5-3, and 3-1
    p_value_5_1 = round(variable_data['G5-G1_p_value'].values[0], 3)
    p_value_5_3 = round(variable_data['G5-G3_p_value'].values[0], 3)
    p_value_3_1 = round(variable_data['G3-G1_p_value'].values[0], 3)

    # Calculate the average of the estimates
    average_estimate = estimates.mean()

    # Create the plot using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(groups, estimates, marker='o', linestyle='-', color='blue', label='Estimate')

    # Plot the confidence intervals
    ax.fill_between(groups, ci_lowers, ci_uppers, color='blue', alpha=0.2, label='95% CI')

    # Add a horizontal line at the covariate average
    ax.axhline(y=average_estimate, color='red', linestyle='--', label=f'Covariate Mean: ({average_estimate:.2f})')

    ax.set_title(f'CLAN of {variable} of {outcome} in {sample} sample')
    ax.set_xlabel('Groups')
    ax.set_ylabel('Covariate Estimates')

    # Force the legend box to be at the top left
    ax.legend(loc='upper left')

    # Annotate the plot with p-values
    textstr = '\n'.join((
        f'p-value (G5-G1): {p_value_5_1:.3f}',
        f'p-value (G5-G3): {p_value_5_3:.3f}',
        f'p-value (G3-G1): {p_value_3_1:.3f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    st.pyplot(fig)

# Streamlit app
st.set_page_config(page_title='CI Plot Viewer', page_icon="🧡", layout="wide")

# File uploader
uploaded_file = "data.xlsx"
uploaded_eval = "eval.xlsx"

clan, blp = st.tabs(["CLAN", "BLP/Gates"])

with clan: 
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        covariates = data['Variable'].unique()
        labels = data.iloc[:, data.columns.get_loc('Variable') + 1].unique()
        label_to_covariate = dict(zip(labels, covariates))
        outcomes = data['Outcome'].unique()
        samples = data['Sample'].unique()
        
        col1, col2 = st.columns(2)
        
        selected_rows = []

        with col1:
            st.subheader("Plot 1")
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                selected_label1 = st.selectbox("Covariate (Plot 1)", labels, key='cov1')
                selected_covariate1 = label_to_covariate[selected_label1]
            with scol2:
                selected_outcome1 = st.selectbox("Outcome (Plot 1)", outcomes, key='out1')
            with scol3:
                selected_sample1 = st.selectbox("Sample (Plot 1)", samples, key='sam1')
            plot_ci(selected_covariate1, selected_outcome1, selected_sample1, data)
            selected_rows.append(data[(data['Variable'] == selected_covariate1) & (data['Outcome'] == selected_outcome1) & (data['Sample'] == selected_sample1)])

            st.subheader("Plot 3")
            scol4, scol5, scol6 = st.columns(3)
            with scol4:
                selected_label2 = st.selectbox("Covariate (Plot 2)", labels, key='cov2')
                selected_covariate2 = label_to_covariate[selected_label2]
            with scol5:
                selected_outcome2 = st.selectbox("Outcome (Plot 2)", outcomes, key='out2')
            with scol6:
                selected_sample2 = st.selectbox("Sample (Plot 2)", samples, key='sam2')
            plot_ci(selected_covariate2, selected_outcome2, selected_sample2, data)
            selected_rows.append(data[(data['Variable'] == selected_covariate2) & (data['Outcome'] == selected_outcome2) & (data['Sample'] == selected_sample2)])

        with col2:
            st.subheader("Plot 2")
            scol7, scol8, scol9 = st.columns(3)
            with scol7:
                selected_label3 = st.selectbox("Covariate (Plot 3)", labels, key='cov3')
                selected_covariate3 = label_to_covariate[selected_label3]
            with scol8:
                selected_outcome3 = st.selectbox("Outcome (Plot 3)", outcomes, key='out3')
            with scol9:
                selected_sample3 = st.selectbox("Sample (Plot 3)", samples, key='sam3')
            plot_ci(selected_covariate3, selected_outcome3, selected_sample3, data)
            selected_rows.append(data[(data['Variable'] == selected_covariate3) & (data['Outcome'] == selected_outcome3) & (data['Sample'] == selected_sample3)])

            st.subheader("Plot 4")
            scol10, scol11, scol12 = st.columns(3)
            with scol10:
                selected_label4 = st.selectbox("Covariate (Plot 4)", labels, key='cov4')
                selected_covariate4 = label_to_covariate[selected_label4]
            with scol11:
                selected_outcome4 = st.selectbox("Outcome (Plot 4)", outcomes, key='out4')
            with scol12:
                selected_sample4 = st.selectbox("Sample (Plot 4)", samples, key='sam4')
            plot_ci(selected_covariate4, selected_outcome4, selected_sample4, data)
            selected_rows.append(data[(data['Variable'] == selected_covariate4) & (data['Outcome'] == selected_outcome4) & (data['Sample'] == selected_sample4)])

        # Concatenate all selected rows and display them
        alld = st.checkbox("Display all Data", value=False)

        selected_data = pd.concat(selected_rows).drop_duplicates()

        if alld == True:
            selected_data = data
        
        st.data_editor(selected_data)

with blp:
    if uploaded_eval is not None:
        evaldata = load_data(uploaded_eval)
        st.data_editor(evaldata)
        st.markdown("* All specifications are with all countries pooled excluding Senegal (AES)")

        
with st.sidebar:
    st.header('Download Data')
    # Add download button for raw data
    with open(uploaded_file, "rb") as file:
        btn = st.download_button(
            label="Download all CLAN data as XLSX",
            data=file,
            file_name='SASP CLAN.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    st.header("Quantile Setting")
    quantiles = st.selectbox("Show calculations based on GATES in:", ("Quintiles", "Terciles"))
    if quantiles == "Terciles":
        st.subheader("Terciles currently calculating. Will be added soon")

