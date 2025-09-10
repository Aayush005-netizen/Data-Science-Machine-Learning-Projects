📖 Project Overview

Floods are among the most destructive natural disasters, impacting millions of lives worldwide. Using historical flood data and environmental indicators such as rainfall, river discharge, soil type, elevation, and land cover, FloodSight builds predictive models to assess flood occurrence.

This project explores multiple machine learning algorithms, compares their performance, and demonstrates how data-driven approaches can support disaster management systems.

🚀 Features

Data preprocessing: cleaning, encoding, scaling.

Visualization of flood trends by land cover & soil type.

Outlier detection using box plots.

Implementation of multiple ML models:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

Histogram-based Gradient Boosting

Neural Networks (MLPClassifier)

Model performance comparison with metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

Environment: Jupyter Notebook / Google Colab

📊 Dataset

The dataset (flood_risk_dataset_india.csv) contains:

Geographical Data: Latitude, Longitude, Elevation

Hydrological Data: Rainfall, River Discharge, Water Level

Climate Data: Temperature, Humidity

Land & Soil Data: Land Cover, Soil Type

Demographic Data: Population Density, Infrastructure index

Target Variable: Flood Occurred (0 = No Flood, 1 = Flood)

⚡ Workflow

Data Preprocessing

Handle missing values and outliers.

Encode categorical variables (Land Cover, Soil Type).

Scale features using MinMaxScaler.

Exploratory Data Analysis (EDA)

Box plots, pie charts, histograms.

Flood occurrences by land cover and soil type.

Model Building & Training

Split dataset (80% train / 20% test).

Train multiple classification models.

Evaluate using performance metrics.

Model Evaluation

Compare algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoost, Neural Network).

Best-performing models identified for future tuning.

📈 Results Summary

Logistic Regression: Accuracy ~ 49–51%

Decision Tree: Accuracy ~ 51%, Recall ~ 90%

Random Forest: Accuracy ~ 50%, overfitting observed (Train = 100%)

Gradient Boosting: Train Accuracy ~ 95%, Test ~ 49%

Hist Gradient Boosting: Train ~ 77%, Test ~ 49%

Neural Network (MLP): Train ~ 62%, Test ~ 50%

👉 Overall, models show potential but require hyperparameter tuning, feature engineering, and more balanced datasets to improve generalization.

🔮 Future Work

Hyperparameter optimization (GridSearchCV, RandomSearch).

Address class imbalance with SMOTE/undersampling.

Integrate live weather APIs for real-time predictions.

Build a web dashboard or mobile app for visualization & alerts.

📌 Getting Started
Installation
git clone https://github.com/Aayush005-netizen/Data-Science-Machine-Learning-Projects.git
cd "Flood Prediction Project"
pip install -r requirements.txt

Running the Notebook

Open the Jupyter/Colab notebook and run all cells:

jupyter notebook flood_prediction.ipynb

🤝 Contributing

Contributions are welcome! Fork the repo, open an issue, or submit a pull request.

Contributions are welcome! Fork the repo, open an issue, or submit a pull request.
