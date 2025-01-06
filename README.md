
# AChE Inhibitor Activity and Potency Prediction Web Application

This Streamlit web application, developed as part of this dissertation, allows users to predict the activity and potency of compounds as potential **AChE inhibitors** (see Figure 26). The application hosts the best models derived from each variant and provides various input options. Users can input individual SMILES strings, structure data files (SDF), or even a list of SMILES from an Excel file. Additionally, users have the option to interactively draw molecules on the Canvas to generate predictions. All inputs are cleaned and standardized using RDKit modules, ensuring consistency before any predictions are made.

The application supports **LIME interpretation** and **atomic contribution mapping** for model interpretability. Specifically:
- **LIME** interpretation is available for models based on **RDKit features** and **AutoML TPOT**.
- **Atomic contribution mapping and visualization** are supported for models that utilize **Graph Convolutional Networks (GCN)**.

## Key Features:
- **Four Modules**: The main application hosts four distinct modules, each corresponding to the best model from a different variant. Each module is designed to handle:
  - Individual SMILES strings.
  - Structure data files (SDF).
  - Lists of SMILES in an Excel file.
  - Interactive structure drawing for activity and potency predictions.
  
- **Data Processing**: The raw SMILES strings and other input formats are cleaned and standardized using RDKit to ensure accurate and consistent data processing.

- **Model Interpretability**: 
  - **LIME interpretation** is available for the RDKit feature-based models derived from **AutoML (TPOT)** and **aggregate modeling**.
  - **Atomic contribution mapping** and visualization are supported for **GCN-based models**, providing insights into how different atoms contribute to the model's predictions.

This tool is designed to assist researchers and practitioners in the field of drug discovery, specifically for compounds aimed at inhibiting AChE, by offering an interactive and transparent way to predict and interpret the biological activity and potency of various compounds.

1. **Clone the Repository**  
Clone the repository to your local machine using the following command:

    ```bash
    git clone https://github.com/bhatnira/AChEI-PredApp.git
    ```

    Alternatively, you can copy and paste the URL into your IDE's source control.

2. **Set Up a Virtual Environment**  
Navigate to the cloned repository folder and create a virtual environment to manage dependencies:

    ```bash
    python3 -m venv env
    ```

    Activate the environment:

    - **On macOS/Linux:**
    
      ```bash
      source ./env/bin/activate
      ```

    - **On Windows:**
    
      ```bash
      .\env\Scripts\activate
      ```

3. **Install Dependencies**  
Once the virtual environment is activated, install the necessary dependencies listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**  
After the dependencies are installed, launch the application by running the following command:

    ```bash
    streamlit run deploy.py
    ```

The app will open in your default browser, and you can start using it to predict AChE inhibitor activity and potency.

The application supports both **Graph Convolutional Network (GCN)-based models** and **descriptor/fingerprint-based models**. LIME-based explanations provide insights into model predictions, helping users understand which features contributed the most to the predicted activity. This tool is designed for researchers and practitioners working on drug discovery and computational chemistry.

Feel free to explore the code, make modifications, and contribute to improving this tool.

For any questions or issues, please open an [issue](https://github.com/bhatnira/AChEI-PredApp/issues) in the repository.
