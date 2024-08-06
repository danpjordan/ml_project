# Machine Learning Sandbox App

## Overview
The Machine Learning Sandbox App is designed to help users process CSV data and perform machine learning tasks with ease. The app allows users to upload CSV files, select target variable and feature from the remaining columns, configure preprocessing parameters, and evaluate models.

Co-Author: [Malik Zekri](https://github.com/TheShadowTiki)

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
1. **Upload a Dataset:**
Click on the "Upload CSV File" button and select a CSV file from your computer.

2. **Select Target Variable:**
Choose the target variable from the dropdown menu. The app will display whether the target variable is discrete or continuous.

3. **Select Features:**
Select the feature columns using the checkboxes. You can use the "Select All" and "Deselect All" buttons to quickly select or deselect all features.

4. **Configure Preprocessing Parameters:**
Adjust the preprocessing parameters such as imputation, removal of invariant features, handling outliers, removal of linearly dependent features (set VIF threshold), and encoding method.

5. **Submit:**
Click the "Submit" button to preprocess the data, train machine learning models, perform inference (classification or regression depending on target variable type), and evaluation. The results will be displayed, and the processed data will be saved to the output directory.

## Application Interface
<div align="center">
    <img src="https://github.com/user-attachments/assets/fbeb28b2-f212-4276-b3e2-8ea8707ccc0b" width="250" alt="Screenshot of the first application window">
    <img src="https://github.com/user-attachments/assets/9882af24-21da-49dd-9d16-2f35f653e6fd" width="700" alt="Screenshot of the evaluation results">
</div>

## Data
This repository includes some open-source CSV files located in the data folder. These files can be used to test the application.

**Included Files:**
1. **data/Employee-Attrition.csv:** Pulled from [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

2. **data/loan_approval_dataset.csv:** Pulled from [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

3. **data/student_performance_data.csv:** Pulled from [📚 Student Performance Dataset 📚](https://www.kaggle.com/datasets/waqi786/student-performance-dataset)

**Usage of Data Files:**
You can use these data files by selecting them when uploading a CSV file in the application. They are provided for testing and demonstration purposes.

## License
This project is licensed under the [MIT] License.
