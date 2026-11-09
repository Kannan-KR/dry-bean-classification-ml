# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple Machine Learning classification algorithms to identify different varieties of dry beans based on their geometric characteristics.

The project uses the Dry Bean Dataset from the UCI Machine Learning Repository. It is a multi-class classification problem involving seven different varieties of dry beans.

Five classification algorithms are implemented and compared using standard evaluation metrics. An interactive Streamlit web application is also developed to provide an interface for uploading test data, selecting classification models and viewing model performance.

### Project Objectives

- Select a suitable public classification dataset.
- Implement five different Machine Learning classification algorithms.
- Evaluate the models using six standard performance metrics.
- Compare the classification performance of all five models.
- Develop an interactive Streamlit application.
- Deploy the application using Streamlit Community Cloud.

## 2. Dataset Description

**Dataset Name:** Dry Bean Dataset

**Dataset Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

### 2.1 Dataset Characteristics

| Description | Details |
|---|---|
| Total instances | 13,611 |
| Number of input features | 16 |
| Target variable | Class |
| Number of target classes | 7 |
| Problem type | Multi-class classification |
| Feature type | Numerical |

### 2.2 Target Classes

The dataset contains seven different varieties of dry beans:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

### 2.3 Input Features

The dataset contains the following 16 numerical input features:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRatio
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. Roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

These features describe the geometric properties and shape characteristics of individual dry beans.

### 2.4 Dataset Reference

Koklu, M. and Ozkan, I. A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture.

**UCI Dataset DOI:** https://doi.org/10.24432/C50S4B

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/Kannan-KR/dry-bean-classification-ml

The repository contains the application source code, model implementation files, dependencies, test dataset and project documentation.

## 4. Machine Learning Models Used

Five supervised Machine Learning classification algorithms were implemented using the Dry Bean Dataset.

### 4.1 Logistic Regression

Logistic Regression is a supervised classification algorithm used to estimate the probability of an observation belonging to different classes. It is used for multi-class classification by estimating the probability of each dry bean variety.

### 4.2 Decision Tree Classifier

Decision Tree is a tree-based classification algorithm that makes predictions by splitting the dataset based on feature values. It uses a sequence of decision rules to classify dry bean varieties.

### 4.3 K-Nearest Neighbor (KNN) Classifier

KNN is a distance-based classification algorithm. It predicts the class of a new observation based on the majority class among its nearest neighbours.

### 4.4 Naive Bayes Classifier

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. Gaussian Naive Bayes is used to estimate class probabilities based on the distribution of numerical features.

### 4.5 Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies decision boundaries to separate different classes. It is used to classify the seven varieties of dry beans.

## 5. Model Evaluation and Comparison

All five classification models are evaluated using six performance metrics.

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC Score:** Measures the model's ability to distinguish between different classes using the Area Under the ROC Curve.
- **Precision:** Measures the proportion of correct predictions among all predictions for a particular class.
- **Recall:** Measures the proportion of actual observations correctly identified by the model.
- **F1 Score:** Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of classification predictions by considering correct and incorrect classifications.

### 5.1 Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| **Support Vector Machine** | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

**Note:** Values are rounded to four decimal places. Precision, recall and F1 use weighted averaging. AUC uses weighted One-vs-Rest averaging. MCC is calculated using the multi-class Matthews correlation coefficient.

### 5.2 Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 92.14% and an AUC score of 0.9934. The model demonstrated consistently high precision, recall and F1 scores, indicating strong classification performance. |
| Decision Tree | Achieved an accuracy of 89.20%. It performed reasonably well but showed lower overall predictive performance than Logistic Regression, KNN and SVM. |
| KNN | Achieved an accuracy of 91.66% and an AUC score of 0.9812. It demonstrated good classification performance, although it was slightly behind Logistic Regression and SVM. |
| Naive Bayes | Achieved an accuracy of 76.39%. It recorded the lowest accuracy and MCC among the five models, indicating comparatively weaker classification performance on this dataset. |
| Support Vector Machine | Achieved the highest accuracy of 92.21%, AUC of 0.9937, precision of 0.9224, recall of 0.9221, F1 score of 0.9222 and MCC of 0.9059. It demonstrated the best overall performance among the five models. |

### 5.3 Overall Winner

**Best Performing Model: Support Vector Machine (SVM)**

Support Vector Machine emerged as the overall best-performing classification model for the Dry Bean Dataset.

It achieved the highest scores across all six evaluation metrics:

| Evaluation Metric | SVM Score |
|---|---:|
| Accuracy | 0.9221 |
| AUC | 0.9937 |
| Precision | 0.9224 |
| Recall | 0.9221 |
| F1 Score | 0.9222 |
| MCC | 0.9059 |

Logistic Regression also demonstrated excellent performance and was very close to SVM across all the evaluation metrics.

Naive Bayes recorded the lowest accuracy and MCC, making it the weakest-performing model in this experiment.

Based on the comparative evaluation, Support Vector Machine is selected as the overall winner for this classification task.

## 6. Streamlit Web Application

An interactive web application has been developed using Streamlit to demonstrate the performance of the five implemented classification models.

**Live Streamlit Application:**

https://dry-bean-classification-ml-gdcbvyetck2ywkuvrh69vk.streamlit.app/

### 6.1 Application Features

1. **Dataset Upload:** Allows users to upload test data in CSV format. If no file is uploaded, the application attempts to use the default `test_data.csv` file.

2. **Model Selection:** Provides a dropdown menu to select one of the five implemented classification models.

3. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.

4. **Confusion Matrix:** Displays a graphical comparison between actual and predicted bean varieties using Seaborn and Matplotlib.

5. **Classification Report:** Displays precision, recall, F1 score and support for individual classes.

6. **Download Predictions:** Allows users to download the prediction results as a CSV file with an additional `Predicted_Class` column.

7. **Model Comparison:** Displays a comparison table containing evaluation results for all five classification models.

### 6.2 Technologies Used in the Application

- Streamlit for the interactive frontend.
- Pandas for dataset handling.
- NumPy for numerical operations.
- Scikit-learn for Machine Learning evaluation metrics.
- Joblib for loading saved models.
- Matplotlib and Seaborn for visualization.

## 7. Project Structure

The application references the following project files:

```text
dry-bean-classification-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── support_vector_machine.pkl
│
└── results/
    └── model_comparison.csv
