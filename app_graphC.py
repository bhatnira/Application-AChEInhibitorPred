import streamlit as st
import os
from rdkit import Chem
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit.Chem.Draw import SimilarityMaps
import tensorflow as tf
tf.random.set_seed(42)

# Streamlit app
st.title('Molecule Predictor and Interpretability')

# Specify the path to your model directory here
model_dir = "/Users/gradstudent/Downloads/graphConv_reg_model_files 2"

# Check if model directory exists
if not os.path.exists(model_dir):
    st.error(f"Model directory '{model_dir}' does not exist.")
    st.stop()

try:
    # Load the model
    n_tasks = 1  # Assuming 1 task for simplicity
    model = dc.models.GraphConvModel(n_tasks, model_dir=model_dir)
    model.restore()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model from '{model_dir}': {str(e)}")
    

# Function to convert SMILES to SDF
def smiles_to_sdf(smiles, sdf_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string"
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    return sdf_path, None

# Function to create dataset
def create_dataset(sdf_path):
    try:
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(), sanitize=True)
        dataset = loader.create_dataset(sdf_path, shard_size=2000)
        return dataset, None
    except Exception as e:
        return None, str(e)

# Function to create fragment dataset
def create_fragment_dataset(sdf_path):
    try:
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True), sanitize=True)
        frag_dataset = loader.create_dataset(sdf_path, shard_size=5000)
        transformer = dc.trans.FlatteningTransformer(frag_dataset)
        frag_dataset = transformer.transform(frag_dataset)
        return frag_dataset, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for whole molecules
def predict_whole_molecules(model, dataset):
    try:
        predictions = np.squeeze(model.predict(dataset))
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)
        predictions_df = pd.DataFrame(predictions[:, 1], index=dataset.ids, columns=["Probability_Class_1"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for fragments
def predict_fragment_dataset(model, frag_dataset):
    try:
        predictions = np.squeeze(model.predict(frag_dataset))[:, 1]
        predictions_df = pd.DataFrame(predictions, index=frag_dataset.ids, columns=["Fragment"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

# Function to visualize contributions
def vis_contribs(mol, df):
    try:
        wt = {}
        for n, atom in enumerate(range(mol.GetNumHeavyAtoms())):
            wt[atom] = df.loc[Chem.MolToSmiles(mol), "Contrib"][n]
        return SimilarityMaps.GetSimilarityMapFromWeights(mol, wt)
    except Exception as e:
        st.warning(f"Error in contribution visualization: {str(e)}")
        return None

# Input type selection: SMILES or Excel File
input_type = st.radio('Choose input type:', ('SMILES', 'Excel File'))

if input_type == 'SMILES':
    # Input SMILES
    input_smiles = st.text_input('Enter SMILES string')

    if st.button('Predict and Show Contribution Map'):
        if input_smiles:
            # Convert SMILES to SDF
            sdf_path = "input_molecule.sdf"
            sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

            if error:
                st.error(f"Error in SMILES to SDF conversion: {error}")
            else:
                try:
                    # Create dataset
                    dataset, error = create_dataset(sdf_path)

                    if error:
                        st.error(f"Error in dataset creation: {error}")
                    else:
                        # Make predictions for whole molecules
                        predictions_whole, error = predict_whole_molecules(model, dataset)

                        if error:
                            st.error(f"Error in predicting whole molecules: {error}")
                        else:
                            # Create fragment dataset
                            frag_dataset, error = create_fragment_dataset(sdf_path)

                            if error:
                                st.error(f"Error in fragment dataset creation: {error}")
                            else:
                                # Make predictions for fragments
                                predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                                if error:
                                    st.error(f"Error in predicting fragments: {error}")
                                else:
                                    # Merge two DataFrames by molecule names
                                    df = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                                    df['Contrib'] = df["Probability_Class_1"] - df["Fragment"]

                                    # Generate molecule from input SMILES
                                    mol = Chem.MolFromSmiles(input_smiles)

                                    # Create maps for the molecule
                                    if mol:
                                        contribution_map = vis_contribs(mol, df)
                                        if contribution_map:
                                            st.write(f"Contribution Map:")
                                            st.write(contribution_map)

                                            # Binary prediction (active/inactive)
                                            threshold = 0.5
                                            prediction_binary = "Active" if predictions_whole.iloc[0, 0] > threshold else "Inactive"
                                            st.write(f"Binary Prediction: {prediction_binary}")

                                            # Prediction probability for class 1
                                            st.write(f"Probability: {predictions_whole.iloc[0, 0]}")
                                        else:
                                            st.warning("Error in generating contribution map.")
                                    else:
                                        st.warning("Unable to generate molecule from input SMILES.")

                                    # Clean up the SDF file after prediction
                                    if os.path.exists(sdf_path):
                                        os.remove(sdf_path)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif input_type == 'Excel File':
    # Handle Excel file input
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
    if uploaded_file is not None:
        try:
            # Load Excel file
            df = pd.read_excel(uploaded_file)

            # Predict for each SMILES in the Excel file
            st.write("Predicting from uploaded Excel file...")
            for index, row in df.iterrows():
                input_smiles = row['SMILES']  # Assuming 'SMILES' is the column name
                #st.write(f"Processing SMILES: {input_smiles}")

                # Convert SMILES to SDF
                sdf_path = f"input_molecule_{index}.sdf"
                sdf_path, error = smiles_to_sdf(input_smiles, sdf_path)

                if error:
                    st.error(f"Error in SMILES to SDF conversion for row {index}: {error}")
                    continue

                # Create dataset
                dataset, error = create_dataset(sdf_path)

                if error:
                    st.error(f"Error in dataset creation for row {index}: {error}")
                    continue

                # Make predictions for whole molecules
                predictions_whole, error = predict_whole_molecules(model, dataset)

                if error:
                    st.error(f"Error in predicting whole molecules for row {index}: {error}")
                    continue

                # Create fragment dataset
                frag_dataset, error = create_fragment_dataset(sdf_path)

                if error:
                    st.error(f"Error in fragment dataset creation for row {index}: {error}")
                    continue

                # Make predictions for fragments
                predictions_frags, error = predict_fragment_dataset(model, frag_dataset)

                if error:
                    st.error(f"Error in predicting fragments for row {index}: {error}")
                    continue

                # Merge two DataFrames by molecule names
                df_result = pd.merge(predictions_frags, predictions_whole, right_index=True, left_index=True)
                df_result['Contrib'] = df_result["Probability_Class_1"] - df_result["Fragment"]

                # Generate molecule from input SMILES
                mol = Chem.MolFromSmiles(input_smiles)

                # Create maps for the molecule
                if mol:
                    contribution_map = vis_contribs(mol, df_result)
                    if contribution_map:
                        st.write(f"------------------------------------------------------------")
                        st.write(f"Prediction and contrib map for input: {input_smiles}:")
                        st.write(contribution_map)

                        # Binary prediction (active/inactive)
                        threshold = 0.5
                        prediction_binary = "Active" if predictions_whole.iloc[0, 0] > threshold else "Inactive"
                        st.write(f"Binary Prediction: {prediction_binary}")

                        # Prediction probability for class 1
                        st.write(f"Prediction Probability for Class 1: {predictions_whole.iloc[0, 0]}")
                    else:
                        st.warning(f"Error in generating contribution map for SMILES {input_smiles}.")
                else:
                    st.warning(f"Unable to generate molecule from input SMILES {input_smiles}.")

                # Clean up the SDF file after prediction
                if os.path.exists(sdf_path):
                    os.remove(sdf_path)

        except Exception as e:
            st.error(f"Error processing uploaded Excel file: {str(e)}")
