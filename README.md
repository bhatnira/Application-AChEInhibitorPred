# AChE Inhibitor Activity Prediction Web Application

This Streamlit-based web application, developed as part of this dissertation, enables users to predict the activity and potency of compounds as potential **AChE inhibitors**. The app leverages advanced machine learning models to provide predictions, hosting the best-performing models derived from different variants. Users can draw molecules directly on the Canvas to generate predictions. All input data is cleaned and standardized before predictions are made. Additionally, the application supports local **LIME** interpretation for GCN-based models and atomic contribution mapping/visualization for descriptor/fingerprint-based models.

To use the application locally, follow the steps below:

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

For any questions or issues, please open an [issue](https://github.com/bhatnira/AChEI-PredApp.git/issues) in the repository.
