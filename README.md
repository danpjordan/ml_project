# Machine Learning Sandbox App

## Overview
The Machine Learning Sandbox App is designed to help users process CSV data and perform machine learning tasks with ease. The app allows users to upload CSV files, select target variable and feature from the remaining columns, configure preprocessing parameters, and evaluate models.

Co-Author: Malik Zekri (https://github.com/TheShadowTiki)

## Installation

1. **Create a virtual environment:**
   ```sh
   conda create -n ml_app python=3.9
   conda activate ml_app

2. **Install Requirements:**
   ```sh
   pip install -r requirements.txt
3. **Run Application:**
   ```sh
   python3 app.py

## Usage
1. **Upload a CSV File:**
Click on the "Upload CSV File" button and select a CSV file from your computer.

2. **Select Target Variable:**
Choose the target variable from the dropdown menu. The app will display whether the target variable is discrete or continuous.

3. **Select Features:**
Select the feature columns using the checkboxes. You can use the "Select All" and "Deselect All" buttons to quickly select or deselect all features.

4. **Configure Preprocessing Parameters:**
Adjust the preprocessing parameters such as imputation, removal of invariant features, handling outliers, VIF threshold, and encoding method.
Submit:

5. Click the **"Submit"** button to preprocess the data and perform machine learning model evaluation. The results will be displayed, and the processed data will be saved to the output directory.

## License
This project is licensed under the [MIT] License.
