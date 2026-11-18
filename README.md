🏠 House Price Prediction

Predicting the price per unit area of a house from its age, location and access to services, using supervised regression in Python.

Machine Learning project (Master 1, Big Data & Data Science), Horizon University. Author: Aycha Belkahla · Instructor: Khouloud Chelbi

Show Image Show Image Show Image

📌 Overview

House prices depend on several factors: how old the property is, how close it is to public transport, what services are nearby and where it is located. This project asks one question:

Can we predict the price of a house per unit area from its characteristics and location?

It is a supervised regression problem built as a complete, reproducible machine learning pipeline:

Exploratory Data Analysis (EDA): understand the data, spot patterns and outliers
Data preparation: feature selection, normalization, train/test split
Modeling: Linear Regression (baseline), Ridge Regression (improved), optional Lasso and Polynomial
Evaluation: MSE and R² metrics, visualizations, interpretation
📊 Dataset

The dataset holds 414 real estate transactions and 8 variables, with no missing values.

Variable Description
X1 transaction date Date of the transaction (numeric format)
X2 house age Age of the house in years (0 to 43.8)
X3 distance to the nearest MRT station Distance to public transport in meters (23 to 6,400+)
X4 number of convenience stores Number of stores within walking distance (0 to 10)
X5 latitude Geographic latitude
X6 longitude Geographic longitude
Y house price of unit area Target: price per unit area (mean ≈ 38)
🔍 Key findings from the EDA
Distance to the nearest MRT station is the strongest negative driver: the farther a house is from public transport, the cheaper it is.
Number of convenience stores has a clear positive relationship with price.
Location (latitude and longitude) strongly influences price.
House age has only a weak negative relationship with price.
Boxplots show few outliers, mostly in the geographic variables. They were kept because they reflect real-world variability.

In short: accessibility and location matter more than the age of the house.

📈 Results

Model results will be added after the modeling and evaluation notebooks are completed.

Model MSE R²
Linear Regression (baseline) to add to add
Ridge Regression to add to add
🗂️ Project structure
house-price-prediction/
│
├── data/
│ ├── realestate.csv # Original dataset
│ ├── realestate_cleaned.csv # Normalized dataset
│ ├── X_train.csv X_test.csv # Training / testing features
│ └── y_train.csv y_test.csv # Training / testing target
│
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory Data Analysis
│ ├── 02_Data_Preparation.ipynb # Feature selection, normalization, train/test split
│ ├── 03_Modeling.ipynb # Baseline and improved regression models
│ └── 04_Evaluation.ipynb # Metrics, visualizations, interpretation
│
├── utils/
│ ├── preprocessing.py # Reusable preprocessing functions
│ └── visualization.py # Reusable plotting and evaluation functions
│
├── requirements.txt
└── README.md
⚙️ Installation

1. Clone the repository

bash
git clone https://github.com/belkahlaaycha-blip/house-price-prediction.git
cd house-price-prediction

2. Create and activate a virtual environment

bash
python -m venv my_ai_env

# Windows

my_ai_env\Scripts\activate

# macOS / Linux

source my_ai_env/bin/activate

3. Install the dependencies

bash
pip install -r requirements.txt

Libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib.

▶️ How to run

Start Jupyter and run the notebooks in order:

bash
jupyter lab
Step Notebook What it does
1 01_EDA.ipynb Loads the raw data, explores it, plots distributions and correlations, exports the cleaned dataset
2 02_Data_Preparation.ipynb Splits features and target, normalizes, creates the train/test sets
3 03_Modeling.ipynb Trains Linear and Ridge Regression (optional: Lasso, Polynomial), exports models and predictions
4 04_Evaluation.ipynb Computes MSE and R², visualizes results, interprets performance
