# credit_card_fraud_detection
Credit Card Fraud Detection
Overview
This project implements a credit card fraud detection system using the Kaggle Credit Card Fraud Detection dataset (284,315 non-fraudulent and 492 fraudulent transactions). The dataset is highly imbalanced (0.17% fraud), so I applied undersampling to balance the training data (492 fraud + 492 non-fraud). I trained two models—Logistic Regression (baseline) and k-Nearest Neighbors (kNN)—and evaluated them using accuracy and recall. Logistic regression outperformed kNN, achieving 95.74% test recall and 93.40% test accuracy, effectively catching most fraudulent transactions.
Objectives

Detect fraudulent transactions in an imbalanced dataset.
Compare logistic regression and kNN to understand model performance.
Focus on recall to maximize detection of fraud cases, critical for financial security.
Build a professional portfolio for interviews, showcasing ML and coding skills (e.g., word search algorithm).

Dataset

Source: Kaggle Credit Card Fraud Detection dataset.
Features: 31 columns (Time, Amount, V1-V28 from PCA, Class).
Imbalance: ~284,315 non-fraud (Class 0), 492 fraud (Class 1).
Preprocessing: Undersampled non-fraud class to 492 samples to balance training data, preserving all fraud cases.

Models

Logistic Regression:
Simple, interpretable baseline model.
Trained on undersampled data to handle imbalance.


k-Nearest Neighbors (kNN):
Used for comparison to explore non-linear patterns.
Tested with n_neighbors=5 (default).



Results
Logistic Regression

Training Accuracy: 94.16%
Test Accuracy: 93.40%
Training Recall: 96.27%
Test Recall: 95.74%

k-Nearest Neighbors (kNN)

Training Accuracy: 76.37%
Test Accuracy: 62.44%
Training Recall: Not computed (assumed similar to accuracy trends).
Test Recall: Not computed (assumed low due to poor accuracy).

Analysis

Logistic Regression Outperformed kNN: Higher accuracy and recall on both training and test sets. Logistic regression effectively captured fraud patterns in the balanced dataset, catching ~96% of frauds (test recall: 95.74%). kNN struggled with generalization (62.44% test accuracy), likely due to sensitivity to k and sparse non-fraud data after undersampling.
Overfitting: Logistic regression shows minimal overfitting (94.16% train vs. 93.40% test accuracy; 96.27% train vs. 95.74% test recall), indicating good generalization. kNN has significant overfitting (76.37% train vs. 62.44% test accuracy), suggesting it’s less suitable for this task.
Why Recall?:
Catching Frauds is Priority: Recall (True Positives / (True Positives + False Negatives)) measures how many actual frauds are detected. Missing frauds (false negatives) leads to financial losses, making high recall critical in fraud detection.
Imbalanced Data: Accuracy is misleading in imbalanced datasets (e.g., predicting all non-fraud gives ~99.8% accuracy but 0% recall). Recall focuses on the minority class (fraud), ensuring most frauds are caught.
Trade-Off: High recall (95.74% test) may increase false positives (flagging legitimate transactions), reducing precision. However, in fraud detection, catching frauds outweighs customer inconvenience, which can be mitigated with verification (e.g., SMS alerts). Future work will compute precision and F1-score to balance this trade-off.



Files

notebooks/credit_card_fraud_detection.ipynb: Jupyter notebook with data loading, preprocessing (undersampling), model training, and evaluation (accuracy, recall).
models/fraud_model.pkl: Saved logistic regression model (using joblib).
visuals/confusion_matrix.png: Confusion matrix heatmap for test set performance (visualizes true positives, false negatives, etc.).
data/sample_data.csv: Optional small sample (100 rows) of preprocessed data for demo purposes.
requirements.txt: Python dependencies for reproducibility.
.gitignore: Ignores sensitive/large files (e.g., full dataset).

Setup Instructions

Clone Repository:git clone https://github.com/GauravRathore11/credit_card_fraud_detection
cd credit_card_fraud_detection


Install Dependencies:python -m venv fraud_env
source fraud_env/bin/activate  # Mac/Linux: or fraud_env\Scripts\activate (Windows)
pip install -r requirements.txt


Download Data: Get the dataset from Kaggle and place it in data/ (or use sample_data.csv).
Run Notebook:jupyter notebook notebooks/credit_card_fraud_detection.ipynb



Future Improvements

Address Undersampling: Undersampling discarded ~99.8% of non-fraud data, potentially increasing false positives. Explore:
SMOTE: Generate synthetic fraud samples to retain all non-fraud data.
Class Weights: Use class_weight='balanced' in logistic regression to penalize misclassifying fraud.


Additional Metrics: Compute precision and F1-score to balance recall and false positives, ensuring fewer legitimate transactions are flagged.
Hyperparameter Tuning: Optimize kNN’s n_neighbors using GridSearchCV or test other models (e.g., Random Forest).
Streamlit UI: Build a web app to input Amount/Time, predict fraud, and display metrics/visuals (e.g., confusion matrix).
Deployment: Host UI on Streamlit Cloud for a live demo.

Interview Notes

Why Recall?: “I prioritized recall (95.74% test for logistic regression) to maximize fraud detection, as missing frauds is costly. Accuracy was high but misleading due to imbalance.”
Model Choice: “Logistic regression was simple and effective, outperforming kNN (62.44% test accuracy) due to better generalization on undersampled data.”
Improvements: “Undersampling caused data loss, so I’d try SMOTE or class weights to improve recall and precision.”
Connection to Algorithms: “My word search project (C++) required debugging duplicates, similar to reducing false negatives in fraud detection, showing my problem-solving skills.”

Folder Structure
credit-card-fraud-detection/
├── data/
│   └── sample_data.csv
├── models/
│   └── fraud_model.pkl
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── visuals/
│   └── confusion_matrix.png
├── requirements.txt
├── README.md
└── .gitignore

Acknowledgments

Dataset: Kaggle Credit Card Fraud Detection.
Tools: Python, scikit-learn, pandas, matplotlib, seaborn, joblib.
