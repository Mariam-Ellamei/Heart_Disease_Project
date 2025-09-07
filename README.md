Heart Disease Prediction Project

This project predicts the likelihood of heart disease using machine learning models trained on the **UCI Heart Disease dataset** (Cleveland subset).  
It includes data preprocessing, feature selection, model training, clustering, hyperparameter tuning, and a Streamlit web app.

---

Project Structure

Heart_Disease_Project/
├── data/
│ ├── processed.cleveland.data # raw dataset
│ └── heart_disease.csv # cleaned dataset
├── scripts/
│ ├── make_csv.py # convert raw -> CSV
│ ├── 01_data_preprocessing.py 
│ ├── 02_pca.py
│ ├── 03_feature_selection.py 
│ ├── 04_supervised_learning.py
│ ├── 05_unsupervised_learning.py
│ ├── 06_hyperparameter_tuning.py
│ ├── 07_export_model.py
├── models/
│ └── heart_disease_model.pkl # final trained model
├── results/
│ ├── correlation_heatmap.png
│ ├── kmeans_scores.png
│ └── pca_scree_plot.png
│ └── preprocessor.pkl
│ └── selected_features.csv
├── ui/
│ └── app.py # Streamlit web application
├── requirements.txt
└── README.md

Setup & Installation

1. Clone this repository or copy the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run preprocessing pipeline (only needed once):
   python scripts/make_csv.py
   python scripts/01_data_preprocessing.py
   python scripts/02_feature_selection.py
   python scripts/03_model_training.py
   python scripts/04_clustering.py
   python scripts/05_hyperparameter_tuning.py

4. Run the Streamlit app:
   streamlit run ui/app.py

Models Used

Logistic Regression → Accuracy ~87%
Random Forest → Accuracy ~82%
SVM → Accuracy ~85%
Random Forest (tuned) → Accuracy ~85%

Streamlit App

The app provides an interactive UI where users enter patient data and receive predictions:

✅ Low risk of heart disease
⚠️ High risk of heart disease

Dataset

Source: UCI Machine Learning Repository – Heart Disease Dataset
Subset: Cleveland Clinic Foundation dataset (303 patients)