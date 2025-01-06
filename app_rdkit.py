import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import traceback
import joblib
import numpy as np
import os
from lime import lime_tabular
import streamlit.components.v1 as components
from streamlit_ketcher import st_ketcher

# Set page config as the very first command
st.set_page_config(
    page_title="Predict Acetylcholinesterase Inhibitory Activity with Interpretation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load custom CSS
def load_css():
    with open("style.css") as f:
        css = f.read()
    components.html(f"<style>{css}</style>", height=0, width=0)

# Function to load Font Awesome icons
def load_fa_icons():
    components.html(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        """,
        height=0, width=0
    )

# Define a function to calculate molecular descriptors
def getMolDescriptors(mol, selected_descriptors, missingVal=None):
    res = {}
    for nm, fn in Descriptors._descList:
        if nm in selected_descriptors:
            try:
                val = fn(mol)
            except Exception:
                traceback.print_exc()
                val = missingVal
            res[nm] = val
    return res

# Load optimized model
def load_optimized_model():
    try:
        model = joblib.load('/Users/gradstudent/Documents/Applications/AChE total_model/bestPipeline_tpot_rdkit_classification.pkl')
        return model
    except Exception as e:
        st.error(f'Error loading optimized model: {e}')
        return None

# Load regression model
def load_regression_model():
    try:
        model = joblib.load('/Users/gradstudent/Documents/Applications/AChE total_model/bestPipeline_tpot_rdkit_Regression.pkl')
        return model
    except Exception as e:
        st.error(f'Error loading regression model: {e}')
        return None

# Function to compute descriptors for a single SMILES input
def compute_descriptors(smiles, selected_descriptors):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = getMolDescriptors(mol, selected_descriptors, missingVal=np.nan)
        descriptor_df = pd.DataFrame([descriptors], columns=selected_descriptors)
        descriptor_df.fillna(0, inplace=True)
        return mol, descriptor_df
    else:
        st.error('Invalid SMILES string.')
        return None, None

# Function to perform prediction and LIME explanation for a single SMILES input
def single_input_prediction(smiles, selected_descriptors, explainer):
    mol, descriptor_df = compute_descriptors(smiles, selected_descriptors)
    if mol is not None:
        classification_model = load_optimized_model()
        regression_model = load_regression_model()
        if classification_model is not None and regression_model is not None:
            try:
                # Classification prediction
                classification_prediction = classification_model.predict(descriptor_df)
                classification_probability = classification_model.predict_proba(descriptor_df)
                
                # Regression prediction
                regression_prediction = regression_model.predict(descriptor_df)
                
                # Generate LIME explanation
                explanation = explainer.explain_instance(descriptor_df.values[0], classification_model.predict_proba, num_features=30)
                return mol, classification_prediction[0], classification_probability[0][1], regression_prediction[0], descriptor_df, explanation
            except Exception as e:
                st.error(f'Error in prediction: {e}')
                return None, None, None, None, None, None
    return None, None, None, None, None, None

# Function to handle drawing input
def handle_drawing_input(explainer, selected_descriptors):
    st.write('Please enter a SMILES string or draw a compound:')
    molecule = st.text_input("Molecule", "")
    smile_code = st_ketcher(molecule)
    st.markdown(f"SMILES from drawing: ``{smile_code}``")

    if st.button('Predict'):
        smiles = smile_code if smile_code else molecule
        if smiles:
            mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(smiles, selected_descriptors, explainer)
            if mol is not None:
                st.write('Activity:', 'potent' if classification_prediction == 1 else 'not potent')
                st.write('Classification Probability:', classification_probability)
                st.write('Predicted IC50(nM):', 10**(regression_prediction))
                st.write('Descriptors:')
                st.write(descriptor_df)
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.image(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True), caption=f'SMILES: {smiles}')
                with col2:                    
                    # Generate download link for LIME explanation
                    st.download_button(
                        label="Download LIME Explanation",
                        data=explanation.as_html(),
                        file_name='lime_explanation.html',
                        mime='text/html'
                    )
        else:
            st.error("Please input a valid SMILES string.")

# Function to handle SMILES input
def handle_smiles_input(explainer, selected_descriptors):
    single_input = st.text_input('Enter a SMILES string:')
    if st.button('Predict'):
        mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(single_input, selected_descriptors, explainer)
        if mol is not None:
            st.write('Activity:', 'potent' if classification_prediction == 1 else 'not potent')
            st.write('Classification Probability:', classification_probability)
            st.write('Predicted IC50(nM):', 10**(regression_prediction)) 
            st.write('Descriptors:')
            st.write(descriptor_df)
            
            col1, col2 = st.columns([2, 3])
            with col1:
                st.image(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True), caption=f'SMILES: {single_input}')
            with col2:
                # Generate download link for LIME explanation
                st.download_button(
                    label="Download LIME Explanation",
                    data=explanation.as_html(),
                    file_name='lime_explanation.html',
                    mime='text/html'
                )

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
def handle_option(option, explainer, selected_descriptors):
    if option == 'Home':
        handle_home_page()
    elif option == 'Single SMILES Input':
        handle_smiles_input(explainer, selected_descriptors)
    elif option == 'Upload SDF File':
        uploaded_sdf_file = st.file_uploader("Upload an SDF file", type=['sdf'], key="sdf_file_uploader")
        if st.button('Predict'):
            sdf_file_prediction(uploaded_sdf_file, selected_descriptors, explainer)
    elif option == 'Upload Excel File':
        uploaded_excel_file = st.file_uploader("Upload an Excel file", type=['xlsx'], key="excel_file_uploader")
        smiles_column = st.text_input("Enter the column name where SMILES are located:")
        if st.button('Predict'):
            excel_file_prediction(uploaded_excel_file, smiles_column, selected_descriptors, explainer)
    elif option == 'Draw Molecule':
        handle_drawing_input(explainer, selected_descriptors)

# Function to handle Excel file prediction
def excel_file_prediction(file, smiles_column, selected_descriptors, explainer):
    if file is not None:
        try:
            df = pd.read_excel(file)
            if smiles_column not in df.columns:
                st.error(f'SMILES column "{smiles_column}" not found in the uploaded file.')
                return
            
            df['Activity'] = np.nan
            df['Classification Probability'] = np.nan
            df['Predicted IC50(nM)'] = np.nan
            
            for index, row in df.iterrows():
                smiles = row[smiles_column]
                mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(smiles, selected_descriptors, explainer)
                if mol is not None:
                    df.at[index, 'Activity'] = 'potent' if classification_prediction == 1 else 'not potent'
                    df.at[index, 'Classification Probability'] = classification_probability
                    df.at[index, 'Predicted IC50(nM)'] = 10**(regression_prediction)
                    for descriptor, value in descriptor_df.iloc[0].items():
                        df.at[index, descriptor] = value
                    
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.image(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True), caption=f"SMILES: {smiles} | Activity: {'potent' if classification_prediction==1 else 'not potent'} | Classification Probability: {classification_probability} | Predicted IC50(nM): {10**(regression_prediction)}")
                    
                    with col2:
                        # Generate download link for LIME explanation
                        st.download_button(
                            label="Download LIME Explanation",
                            data=explanation.as_html(),
                            file_name=f'lime_explanation_{index}.html',
                            mime='text/html'
                        )
            
            st.write(df)
            
        except Exception as e:
            st.error(f'Error loading data: {e}')
    else:
        st.warning('Please upload a file containing SMILES strings.')

# Function to handle SDF file prediction
def sdf_file_prediction(file, selected_descriptors, explainer):
    if file is not None:
        try:
            # Save the uploaded SDF file temporarily
            with open("temp.sdf", "wb") as f:
                f.write(file.getvalue())
            
            suppl = Chem.SDMolSupplier("temp.sdf")
            if suppl is None:
                st.error('Failed to load SDF file.')
                return
            
            for mol in suppl:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(smiles, selected_descriptors, explainer)
                    if mol is not None:
                        st.write('Activity:', 'potent' if classification_prediction == 1 else 'not potent')
                        st.write('Classification Probability:', classification_probability)
                        st.write('Predicted IC50(nM):', 10**(regression_prediction))
                        st.write('Descriptors:')
                        st.write(descriptor_df)
                        
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            st.image(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True), caption=f'SMILES: {smiles} | Activity: {classification_prediction} | Classification Probability: {classification_probability} | Predicted IC50(nM): {10**(regression_prediction)}')
                        
                        with col2:
                            # Generate download link for LIME explanation
                            st.download_button(
                                label="Download LIME Explanation",
                                data=explanation.as_html(),
                                file_name=f'lime_explanation_{smiles}.html',
                                mime='text/html'
                            )
            
        except Exception as e:
            st.error(f'Error processing SDF file: {e}')
        finally:
            # Delete the temporary file
            os.remove("temp.sdf")
    else:
        st.warning('Please upload an SDF file.')

if __name__ == '__main__':
    # Load Font Awesome icons
    load_fa_icons()
    
    # Load training data to initialize the LIME explainer
    train_df = pd.read_pickle('/Users/gradstudent/Documents/Applications/AChE total_model/X_train_circular.pkl')  # Replace with actual path
    
    # Define class labels
    class_names = {0: '0', 1: '1'}
    
    explainer = lime_tabular.LimeTabularExplainer(train_df.values,
                                                  feature_names=train_df.columns.tolist(),
                                                  class_names=class_names.values(),  # Use class labels here
                                                  discretize_continuous=True)
    
    st.title('Predict Acetylcholinesterase Inhibitory Activity with Interpretation')
    
    # Display table output
    st.write('Please enter or upload SMILES data to generate predictions.')
    st.write('')
    
    st.sidebar.title('Options')
    
    # Sidebar with Font Awesome icons
    option = st.sidebar.selectbox(
        'Choose an option:',
        [
            'Home',
            'Single SMILES Input',
            'Upload SDF File',
            'Upload Excel File',
            'Draw Molecule'
        ]
    )
    
    handle_option(option, explainer, selected_descriptors=['MaxEStateIndex',
 'MinEStateIndex',
 'MaxAbsEStateIndex',
 'MinAbsEStateIndex',
 'qed',
 'MolWt',
 'HeavyAtomMolWt',
 'ExactMolWt',
 'NumValenceElectrons',
 'FpDensityMorgan1',
 'FpDensityMorgan2',
 'FpDensityMorgan3',
 'BalabanJ',
 'BertzCT',
 'Chi0',
 'Chi0n',
 'Chi0v',
 'Chi1',
 'Chi1n',
 'Chi1v',
 'Chi2n',
 'Chi2v',
 'Chi3n',
 'Chi3v',
 'Chi4n',
 'Chi4v',
 'HallKierAlpha',
 'Ipc',
 'Kappa1',
 'Kappa2',
 'Kappa3',
 'LabuteASA',
 'PEOE_VSA1',
 'PEOE_VSA10',
 'PEOE_VSA11',
 'PEOE_VSA12',
 'PEOE_VSA13',
 'PEOE_VSA14',
 'PEOE_VSA2',
 'PEOE_VSA3',
 'PEOE_VSA4',
 'PEOE_VSA5',
 'PEOE_VSA6',
 'PEOE_VSA7',
 'PEOE_VSA8',
 'PEOE_VSA9',
 'SMR_VSA1',
 'SMR_VSA10',
 'SMR_VSA2',
 'SMR_VSA3',
 'SMR_VSA4',
 'SMR_VSA5',
 'SMR_VSA6',
 'SMR_VSA7',
 'SMR_VSA9',
 'SlogP_VSA1',
 'SlogP_VSA10',
 'SlogP_VSA11',
 'SlogP_VSA12',
 'SlogP_VSA2',
 'SlogP_VSA3',
 'SlogP_VSA4',
 'SlogP_VSA5',
 'SlogP_VSA6',
 'SlogP_VSA7',
 'SlogP_VSA8',
 'TPSA',
 'EState_VSA1',
 'EState_VSA10',
 'EState_VSA11',
 'EState_VSA2',
 'EState_VSA3',
 'EState_VSA4',
 'EState_VSA5',
 'EState_VSA6',
 'EState_VSA7',
 'EState_VSA8',
 'EState_VSA9',
 'VSA_EState1',
 'VSA_EState10',
 'VSA_EState2',
 'VSA_EState3',
 'VSA_EState4',
 'VSA_EState5',
 'VSA_EState6',
 'VSA_EState7',
 'VSA_EState8',
 'VSA_EState9',
 'FractionCSP3',
 'HeavyAtomCount',
 'NHOHCount',
 'NOCount',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumAromaticRings',
 'NumHAcceptors',
 'NumHDonors',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'RingCount',
 'MolLogP',
 'MolMR',
 'fr_Al_COO',
 'fr_Al_OH',
 'fr_Al_OH_noTert',
 'fr_ArN',
 'fr_Ar_COO',
 'fr_Ar_N',
 'fr_Ar_NH',
 'fr_Ar_OH',
 'fr_COO',
 'fr_COO2',
 'fr_C_O',
 'fr_C_O_noCOO',
 'fr_C_S',
 'fr_HOCCN',
 'fr_Imine',
 'fr_NH0',
 'fr_NH1',
 'fr_NH2',
 'fr_N_O',
 'fr_Ndealkylation1',
 'fr_Ndealkylation2',
 'fr_Nhpyrrole',
 'fr_SH',
 'fr_aldehyde',
 'fr_alkyl_carbamate',
 'fr_alkyl_halide',
 'fr_allylic_oxid',
 'fr_amide',
 'fr_amidine',
 'fr_aniline',
 'fr_aryl_methyl',
 'fr_azide',
 'fr_benzene',
 'fr_bicyclic',
 'fr_dihydropyridine',
 'fr_epoxide',
 'fr_ester',
 'fr_ether',
 'fr_furan',
 'fr_guanido',
 'fr_halogen',
 'fr_hdrzine',
 'fr_hdrzone',
 'fr_imidazole',
 'fr_imide',
 'fr_ketone',
 'fr_ketone_Topliss',
 'fr_lactam',
 'fr_lactone',
 'fr_methoxy',
 'fr_morpholine',
 'fr_nitrile',
 'fr_nitro',
 'fr_nitro_arom',
 'fr_nitro_arom_nonortho',
 'fr_oxazole',
 'fr_oxime',
 'fr_para_hydroxylation',
 'fr_phenol',
 'fr_phenol_noOrthoHbond',
 'fr_phos_acid',
 'fr_phos_ester',
 'fr_piperdine',
 'fr_piperzine',
 'fr_priamide',
 'fr_pyridine',
 'fr_quatN',
 'fr_sulfide',
 'fr_sulfonamd',
 'fr_sulfone',
 'fr_term_acetylene',
 'fr_tetrazole',
 'fr_thiazole',
 'fr_thiophene',
 'fr_unbrch_alkane',
 'fr_urea']
)  # Specify selected descriptors here
