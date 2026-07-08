# 🌸 Iris Flower Classification Portal

Welcome to the **Iris Flower Classification Portal**! This is an interactive machine learning web application designed to explore the famous Iris Flower dataset, train classification models in real time with custom hyperparameters, and perform predictions on user-input features with real-time feedback.

---

## 🚀 Features

- 📊 **Dashboard & Analytics (EDA)**:
  - Interactive table representing raw data.
  - Descriptive statistics (mean, standard deviation, range, etc.).
  - Interactive 2D scatter plots of sepal/petal dimensions color-coded by species.
  - Distribution histograms with boxplots for each feature.
  - Dynamic 3D feature space visualization.
  - Pearson correlation heatmap of features.
  
- 🧠 **Interactive Model Trainer (Machine Learning Lab)**:
  - Supports 5 classification algorithms:
    - **Random Forest**
    - **Support Vector Machine (SVM)**
    - **K-Nearest Neighbors (KNN)**
    - **Logistic Regression**
    - **Decision Tree**
  - Interactive hyperparameter tuning (e.g., number of estimators, depth, regularizers, kernels, neighbor count).
  - Configurable train-test splits and random seed reproducibility.
  - Real-time training metrics (Accuracy, F1-Score, Precision, Recall).
  - Beautifully styled Confusion Matrix.
  - Feature Importance charts for supported algorithms.

- 🔮 **Real-Time Predictor (Inference)**:
  - User-friendly sliders to input custom Sepal and Petal dimensions.
  - Real-time prediction runs as soon as input dimensions change.
  - Visual classification badges (color-coded by species).
  - Multi-class probability distributions.
  - Visual illustrations of the predicted specimen.

---

## 📁 Repository Structure

```
Iris Flower Classification/
├── assets/                    # Image assets for the three species
│   ├── setosa.png
│   ├── versicolor.png
│   └── virginica.png
├── app.py                     # Main Streamlit web application
├── model.py                   # Machine learning training & inference logic
├── Iris.csv                   # The raw Iris dataset
├── requirements.txt           # Project dependencies
├── install.bat                # Windows Batch script to install requirements
├── run.bat                    # Windows Batch script to run the Streamlit app
├── uninstall.bat              # Windows Batch script to uninstall requirements
└── README.md                  # Project documentation
```

---

## 🛠️ Setup & How to Run

This project runs on Python (version 3.8+ recommended).

### Windows Setup (Quick Start)
1. **Install Dependencies**: Double-click `install.bat`. This will verify Python and install all libraries from `requirements.txt`.
2. **Launch the Application**: Double-click `run.bat`. The web application will launch, and your browser should automatically open the dashboard (typically at `http://localhost:8501`).
3. **Clean Up**: If you wish to uninstall the dependencies later, double-click `uninstall.bat`.

### Manual CLI Setup
If you prefer using the command line:
1. Open terminal and navigate to the project directory:
   ```bash
   cd "Iris Flower Classification"
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Machine Learning Algorithms

- **Random Forest**: An ensemble learning method that constructs a multitude of decision trees and outputs the class that is the mode of the classes. Excellent for handling non-linear boundaries.
- **SVM (Support Vector Machine)**: Effective in high-dimensional spaces. Uses support vectors and kernel functions (like Radial Basis Function or Linear) to maximize the classification margin.
- **K-Nearest Neighbors**: A simple instance-based learning algorithm that classifies a sample based on the majority vote of its neighbors.
- **Logistic Regression**: A linear model that estimates probabilities using a logistic function. Highly efficient and interpretable.
- **Decision Tree**: Breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. Highly interpretable.

---
Created as part of the **CodeAlpha** Internship Project.
