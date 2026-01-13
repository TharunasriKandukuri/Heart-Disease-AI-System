Heart Disease Prediction – End-to-End AI System  
Multi-Stage Intelligent Prediction & Insight Engine

Problem Statement
The objective of this project is to design and implement an end-to-end AI/ML system that predicts health risk (Heart Disease) using structured medical data, and enhances insights using unstructured text data through text classification.

This project is built with production thinking, covering:
- Data understanding
- Feature engineering
- Model development
- Error analysis
- Explainability
- Deployment architecture



Datasets Used

Structured Dataset
Source:Public Heart Disease Dataset
Features Include:
  - Age
  - Sex
  - Chest pain type
  - Blood pressure
  - Cholesterol
  - ECG results
  - Max heart rate
  - Exercise-induced angina
  - ST depression
  - Number of vessels
  - Thallium
  - Target Variable:Heart Disease (Binary Classification)

Unstructured Dataset 
- Source:SMS Spam Collection Dataset
- Data Type:Raw text messages
- Labels:Spam / Ham

Structured Data
- Schema and feature type analysis
- Missing value handling using statistical methods (median)
- Class imbalance analysis
- Outlier awareness (IQR / Z-score concepts applied)

Unstructured Text Data
- Lowercasing
- Removing special characters
- Stopword removal using NLTK
- Cleaned textual corpus generation

Exploratory Data Analysis (EDA)
- Target distribution analysis
- Feature relationships
- Correlation insights
- Class imbalance observation

Feature Engineering

Structured Data
- Feature transformation
- Feature selection using correlation analysis
- Importance-based feature understanding

Text Data
- Tokenization
- Stopword removal
- TF-IDF Vectorization

Models Developed

Structured ML Models
1. Baseline Model
   - Logistic Regression
2. Tree-Based Model
   - Random Forest
3. Advanced Model
   - Neural Network (MLP)

Text Classification Model
- TF-IDF + Classification Model (Spam vs Ham)

Model Optimization & Tuning
- Hyperparameter tuning using GridSearchCV
- Cross-validation applied
- Overfitting and underfitting addressed
- Model performance comparison across algorithms

 Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation scores

Error Analysis & Root Cause Analysis (RCA)
- Identification of:
  - False Positives
  - False Negatives
- Explanation of:
  - Why misclassifications occurred
  - Patterns causing prediction errors
- Suggested corrective strategies:
  - Feature enrichment
  - Better class balancing
  - Threshold tuning

 Model Explainability
- Feature importance analysis
- Interpretation of model decisions
- Explainability insights for medical decision support

 Deployment & System Design (Conceptual)

Proposed Deployment Stack
- Backend:Flask / FastAPI
- Model Serving:Pickle / Joblib models
- Cloud: AWS (EC2 / S3 / API Gateway)

System Flow
1. User inputs data via API
2. Data preprocessing pipeline
3. Model inference
4. Prediction response returned
5. Logs stored for monitoring

Performance & Scalability Considerations
- Batch vs real-time prediction handling
- Model versioning strategy
- Monitoring for data drift
- Scalability for large datasets
- CI/CD integration using GitHub

 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- NLTK
- Jupyter Notebook
- GitHub


Author
Tharuna Sri Kandukuri
