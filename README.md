a. Problem Statement
The objective of this project is to predict whether a digital advertisement campaign achieves high Return on Ad Spend (ROAS) based on historical advertising performance data.
This is formulated as a binary classification problem, where campaigns with ROAS greater than 1 are labeled as High ROAS (1) and others as Low ROAS (0).
Accurate prediction of high-performing advertisements helps organizations optimize marketing budgets, improve campaign efficiency, and maximize overall revenue.
________________________________________

b. Dataset Description
The dataset used in this project is the Global Ads Performance Dataset, obtained from a public repository.
Dataset Characteristics:
•	Type: Binary Classification
•	Target Variable: High_ROAS
•	Number of instances: 1200
•	Number of features:  14
Key Preprocessing Steps:
•	Created binary target variable High_ROAS from ROAS
•	Removed irrelevant columns such as date
•	Handled missing values by removing incomplete rows
•	Encoded categorical variables using Label Encoding
•	Applied feature scaling for Logistic Regression and KNN models
________________________________________

c. Models Used and Evaluation Metrics
The following six machine learning classification models were implemented and evaluated on the same dataset:
1.	Logistic Regression
2.	Decision Tree Classifier
3.	K-Nearest Neighbors (KNN)
4.	Naive Bayes (Gaussian)
5.	Random Forest (Ensemble Model)
6.	XGBoost (Ensemble Model)
Each model was evaluated using the following metrics:
•	Accuracy
•	AUC Score
•	Precision
•	Recall
•	F1 Score
•	Matthews Correlation Coefficient (MCC)
________________________________________

 Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9550 | 0.9814 | 0.9594 | 0.9934 | 0.9761 | 0.6252 |
| Decision Tree | 0.9933 | 0.9694 | 0.9952 | 0.9976 | 0.9964 | 0.9517 |
| KNN | 0.9356 | 0.9245 | 0.9408 | 0.9928 | 0.9661 | 0.3903 |
| Naive Bayes | 0.9156 | 0.9211 | 0.9743 | 0.9333 | 0.9533 | 0.5245 |
| Random Forest (Ensemble) | 0.9917 | 0.9954 | 0.9928 | 0.9982 | 0.9955 | 0.9391 |
| XGBoost (Ensemble) | 0.9961 | 0.9980 | 0.9964 | 0.9994 | 0.9979 | 0.9719 |


         Observations on Model Performance

| ML Model | Observation about model performance |
|---------|------------------------------------|
| Logistic Regression | Achieved strong baseline performance with 95.50% accuracy and high AUC. However, the moderate MCC value indicates limited robustness under class imbalance due to its linear nature. |
| Decision Tree | Delivered very high accuracy and MCC, showing strong classification capability. However, such high performance suggests a potential risk of overfitting. |
| KNN | Demonstrated high recall but comparatively low MCC, indicating sensitivity to class imbalance and dependence on distance-based learning. |
| Naive Bayes | Provided reasonable performance with fast computation, but its independence assumption limits predictive accuracy on complex feature interactions. |
| Random Forest (Ensemble) | Achieved excellent and balanced performance across metrics, benefiting from ensemble learning and reduced overfitting. |
| XGBoost (Ensemble) | Delivered the best overall results with the highest accuracy, AUC, and MCC, effectively capturing nonlinear relationships through boosting. |

Conclusion

Among all evaluated models, ensemble methods outperform individual classifiers due to their ability to model complex feature interactions and reduce overfitting.
XGBoost demonstrates the most balanced and superior performance across all evaluation metrics, making it the most suitable model for predicting high ROAS advertisement campaigns.

