import streamlit as st
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
import joblib
from lime import lime_tabular
import streamlit.components.v1 as components
import tempfile
import os

# Set page config as the very first command
st.set_page_config(
    page_title="Predict Acetylcholinesterase Inhibitory Activity with Interpretation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load Font Awesome icons
def load_fa_icons():
    components.html(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        """,
        height=0, width=0
    )

# Function to generate circular fingerprints
def get_circular_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        fingerprint = featurizer.featurize([mol])
        return fingerprint[0]
    else:
        st.error('Invalid SMILES string.')
        return None

# Load optimized model
@st.cache_data
def load_optimized_model():
    try:
        class_model = joblib.load('/Users/gradstudent/Documents/Applications/AChE total_model/bestPipeline_tpot_circularfingerprint_classification.pkl')
        return class_model
    except Exception as e:
        st.error(f'Error loading optimized model: {e}')
        return None

# Load regression model
@st.cache_data
def load_regression_model():
    try:
        reg_model = joblib.load('/Users/gradstudent/Documents/Applications/AChE total_model/best_model_aggregrate_circular.pkl')
        return reg_model
    except Exception as e:
        st.error(f'Error loading regression model: {e}')
        return None

# Load training data
@st.cache_data
def load_training_data():
    try:
        training_data = pd.read_pickle('/Users/gradstudent/Documents/Applications/AChE total_model/X_train_circular.pkl')
        return training_data
    except Exception as e:
        st.error(f'Error loading training data: {e}')
        return None

# Function to perform prediction and LIME explanation for a single SMILES input
def single_input_prediction(smiles, explainer):
    fingerprint = get_circular_fingerprint(smiles)
    if fingerprint is not None:
        descriptor_df = pd.DataFrame([fingerprint])
        
        classification_model = load_optimized_model()
        regression_model = load_regression_model()
        if classification_model is not None and regression_model is not None:
            try:
                classification_prediction = classification_model.predict(descriptor_df)
                classification_probability = classification_model.predict_proba(descriptor_df)
                
                regression_prediction = regression_model.predict(descriptor_df)
                
                explanation = explainer.explain_instance(descriptor_df.values[0], classification_model.predict_proba, num_features=30)
                return fingerprint, descriptor_df, classification_prediction[0], classification_probability[0][1], regression_prediction[0], explanation
            except Exception as e:
                st.error(f'Error in prediction: {e}')
                return None, None, None, None, None, None
    return None, None, None, None, None, None

# Function to display prediction results consistently
def display_prediction_results(smiles, fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation):
    st.write('**SMILES:**', smiles)
    st.write('**Activity:**', 'potent' if classification_prediction == 1 else 'not potent')
    st.write('**Classification Probability:**', classification_probability)
    st.write('**Predicted IC50(nM):**', np.power(10, regression_prediction))
    
    col1, col2 = st.columns([2, 3])
    with col1:
        mol = Chem.MolFromSmiles(smiles)
        st.image(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True), caption=f'SMILES: {smiles}')
    
    with col2:
        st.download_button(
            label="Download LIME Explanation",
            data=explanation.as_html(),
            file_name='lime_explanation.html',
            mime='text/html'
        )

# Function to handle drawing input
def handle_drawing_input(explainer):
    st.write('Please enter a SMILES string or draw a compound:')
    molecule = st.text_input("Molecule", "")
    smile_code = st_ketcher(molecule)
    st.markdown(f"SMILES from drawing: `{smile_code}`")

    if st.button('Predict'):
        smiles = smile_code if smile_code else molecule
        if smiles:
            fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation = single_input_prediction(smiles, explainer)
            if fingerprint is not None:
                display_prediction_results(smiles, fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation)
        else:
            st.error("Please input a valid SMILES string.")

# Function to handle SMILES input
def handle_smiles_input(explainer):
    single_input = st.text_input('Enter a SMILES string:')
    if st.button('Predict'):
        fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation = single_input_prediction(single_input, explainer)
        if fingerprint is not None:
            display_prediction_results(single_input, fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation)

# Function to handle the home page
def handle_home_page():
    st.write('Welcome to the Molecule Activity Prediction App!')
    st.write('This app predicts the acetylcholinesterase inhibitory activity of molecules based on their SMILES representation.')
    st.write('You can choose to:')
    st.write('- Enter a SMILES string')
    st.write('- Upload an SDF file')
    st.write('- Upload an Excel file with SMILES strings')
    st.write('- Draw a molecule')
    st.write('')
    st.write('Use the sidebar to navigate to different functionalities of the app.')

# Function to handle the selected option
def handle_option(option, explainer):
    if option == 'Home':
        handle_home_page()
    elif option == 'Single SMILES Input':
        handle_smiles_input(explainer)
    elif option == 'Upload SDF File':
        uploaded_sdf_file = st.file_uploader("Upload an SDF file", type=['sdf'], key="sdf_file_uploader")
        if uploaded_sdf_file:
            if st.button('Predict'):
                sdf_file_prediction(uploaded_sdf_file, explainer)
    elif option == 'Upload Excel File':
        uploaded_excel_file = st.file_uploader("Upload an Excel file", type=['xlsx'], key="excel_file_uploader")
        smiles_column = st.text_input("Enter the column name where SMILES are located:")
        if uploaded_excel_file and smiles_column:
            if st.button('Predict'):
                excel_file_prediction(uploaded_excel_file, smiles_column, explainer)
    elif option == 'Draw Molecule':
        handle_drawing_input(explainer)

# Function to handle Excel file prediction
def excel_file_prediction(file, smiles_column, explainer):
    if file is not None:
        try:
            df = pd.read_excel(file)
            if smiles_column not in df.columns:
                st.error(f'SMILES column "{smiles_column}" not found in the uploaded file.')
                return
            
            # Add columns for results
            df['Activity'] = np.nan
            df['Classification Probability'] = np.nan
            df['Predicted IC50(nM)'] = np.nan

            # Predict results for each row
            for index, row in df.iterrows():
                smiles = row[smiles_column]
                fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation = single_input_prediction(smiles, explainer)
                if fingerprint is not None:
                    df.at[index, 'Activity'] = 'potent' if classification_prediction == 1 else 'not potent'
                    df.at[index, 'Classification Probability'] = classification_probability
                    df.at[index, 'Predicted IC50(nM)'] = np.power(10, regression_prediction)
                    for descriptor, value in descriptor_df.iloc[0].items():
                        df.at[index, descriptor] = value

                    # Optionally display image and LIME explanation
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        mol = Chem.MolFromSmiles(smiles)
                        st.image(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True), caption=f"SMILES: {smiles} | Activity: {'potent' if classification_prediction == 1 else 'not potent'} | Classification Probability: {classification_probability} | Predicted IC50(nM): {np.power(10, regression_prediction)}")
                    
                    with col2:
                        st.download_button(
                            label="Download LIME Explanation",
                            data=explanation.as_html(),
                            file_name=f'lime_explanation_{index}.html',
                            mime='text/html'
                        )

            st.write(df)
            df.to_excel('predictions.xlsx', index=False)
            st.download_button(
                label="Download Predictions",
                data=open('predictions.xlsx', 'rb').read(),
                file_name='predictions.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            st.error(f'Error processing Excel file: {e}')

# Function to handle SDF file prediction
def sdf_file_prediction(file, explainer):
    if file is not None:
        try:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmpfile:
                tmpfile.write(file.read())
                temp_file_path = tmpfile.name

            # Use Chem.SDMolSupplier with the path to the temporary file
            sdf_supplier = Chem.SDMolSupplier(temp_file_path)
            results = []
            for mol in sdf_supplier:
                if mol:
                    smiles = Chem.MolToSmiles(mol)
                    fingerprint, descriptor_df, classification_prediction, classification_probability, regression_prediction, explanation = single_input_prediction(smiles, explainer)
                    if fingerprint is not None:
                        result = {
                            'SMILES': smiles,
                            'Activity': 'potent' if classification_prediction == 1 else 'not potent',
                            'Classification Probability': classification_probability,
                            'Predicted IC50(nM)': np.power(10, regression_prediction)
                        }
                        results.append(result)

            results_df = pd.DataFrame(results)
            st.write(results_df)
            
            # Save results to Excel and offer for download
            results_df.to_excel('sdf_predictions.xlsx', index=False)
            st.download_button(
                label="Download Predictions",
                data=open('sdf_predictions.xlsx', 'rb').read(),
                file_name='sdf_predictions.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            st.error(f'Error processing SDF file: {e}')
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# Main function
def main():
    load_fa_icons()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=load_training_data().values,
        feature_names=load_training_data().columns.tolist(),
        class_names=['not potent', 'potent'],
        mode='classification'
    )
    
    option = st.sidebar.selectbox(
        'Select an option',
        ['Home', 'Single SMILES Input', 'Upload SDF File', 'Upload Excel File', 'Draw Molecule']
    )
    
    handle_option(option, explainer)

if __name__ == "__main__":
    main()
