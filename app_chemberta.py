import streamlit as st
import pandas as pd
from simpletransformers.classification import ClassificationModel
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import os
import streamlit.components.v1 as components
from streamlit_ketcher import st_ketcher
import torch
import torch.nn.functional as F

# Path to the directory where the model was saved
saved_model_path = "/Users/gradstudent/Documents/Applications/AChE total_model/checkpoint-2000"

# Load the model
loaded_model = ClassificationModel('roberta', saved_model_path, use_cuda=False)

# Function to predict toxicity using CHEMBERTA model
def predict_chemberta(smiles):
    predictions, raw_outputs = loaded_model.predict([smiles])
    
    # Extract logits from raw outputs (assuming binary classification)
    logits = raw_outputs[0]
    
    # Compute the probability of class 1 (toxic)
    prob_toxic = F.sigmoid(torch.tensor(logits[1])).item()
    
    return predictions[0], logits, prob_toxic

# Function to handle drawing input
def handle_drawing_input():
    st.write('Please enter a SMILES string or draw a compound:')
    molecule = st.text_input("Molecule", "")
    smile_code = st_ketcher(molecule)
    st.markdown(f"SMILES from drawing: ``{smile_code}``")

    if st.button('Predict'):
        smiles = smile_code if smile_code else molecule
        if smiles:
            prediction, logits, prob_toxic = predict_chemberta(smiles)
            if prediction == 1:
                st.write('Prediction: Potent')
            else:
                st.write('Prediction: Not-potent')
            st.write('Logits: Logit inactive = {:.4f}, Logit active  = {:.4f}'.format(logits[0], logits[1]))
            st.write('Computed Prediction Probability: {:.4f}'.format(prob_toxic))
        else:
            st.error("Please input a valid SMILES string.")

# Function to handle SMILES input
def handle_smiles_input():
    smiles_input = st.text_input('Enter a SMILES string:')
    if st.button('Predict'):
        prediction, logits, prob_toxic = predict_chemberta(smiles_input)
        if prediction == 1:
            st.write('Prediction: Toxic')
        else:
            st.write('Prediction: Non-Toxic')
        
        st.write('Logits: Logit 0 = {:.4f}, Logit 1 = {:.4f}'.format(logits[0], logits[1]))
        st.write('Probability of Toxic: {:.4f}'.format(prob_toxic))

# Function to handle SDF file upload
def handle_sdf_upload():
    uploaded_file = st.file_uploader("Upload an SDF file", type=['sdf'])
    if st.button('Predict'):
        if uploaded_file is not None:
            try:
                # Save the uploaded SDF file temporarily
                with open("temp.sdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                suppl = Chem.SDMolSupplier("temp.sdf")
                if suppl is not None:
                    for mol in suppl:
                        if mol is not None:
                            smiles = Chem.MolToSmiles(mol)
                            prediction, logits, prob_toxic = predict_chemberta(smiles)
                            if prediction == 1:
                                st.write('Prediction: Toxic')
                            else:
                                st.write('Prediction: Non-Toxic')
                            
                            st.write('Logits: Logit 0 = {:.4f}, Logit 1 = {:.4f}'.format(logits[0], logits[1]))
                            st.write('Probability of Toxic: {:.4f}'.format(prob_toxic))
                else:
                    st.error('Failed to load SDF file.')
            except Exception as e:
                st.error(f'Error processing SDF file: {e}')
            finally:
                # Delete the temporary file
                os.remove("temp.sdf")
        else:
            st.warning('Please upload an SDF file.')

# Function to handle Excel file upload
def handle_excel_upload():
    uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx'])
    if uploaded_file is not None:
        smiles_column = st.text_input("Enter the column name where SMILES are located:")
        if st.button('Predict'):
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                if smiles_column not in df.columns:
                    st.error(f'SMILES column "{smiles_column}" not found in the uploaded file.')
                    return
                
                predictions = []
                logits_list = []
                prob_toxic_list = []
                for smiles in df[smiles_column].dropna():
                    prediction, logits, prob_toxic = predict_chemberta(smiles)
                    predictions.append(prediction)
                    logits_list.append(logits)
                    prob_toxic_list.append(prob_toxic)
                
                df['Prediction'] = predictions
                df['Logit 0'] = [logits[0] for logits in logits_list]
                df['Logit 1'] = [logits[1] for logits in logits_list]
                df['Probability of Toxic'] = prob_toxic_list
                st.write(df)
                
            except Exception as e:
                st.error(f'Error loading data: {e}')
    else:
        st.warning('Please upload an Excel file.')

# Main function to handle user input and display predictions
def main():
    st.title('CHEMBERTA Predictor')
    st.write("Choose an option below to predict:")
    option = st.selectbox('Prediction Options', ['Single SMILES Input', 'Upload SDF File', 'Upload Excel File', 'Draw Molecule'])

    if option == 'Single SMILES Input':
        handle_smiles_input()
    elif option == 'Upload SDF File':
        handle_sdf_upload()
    elif option == 'Upload Excel File':
        handle_excel_upload()
    elif option == 'Draw Molecule':
        handle_drawing_input()

if __name__ == '__main__':
    main()
