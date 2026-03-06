# Bankruptcy Prediction Model – Poland & Taiwan Companies  

This repository contains my **fifth project** from the **Applied Data Science Lab at WorldQuant University**, where I developed a **machine learning model to predict bankruptcy** for companies based on real-world financial data from Poland (for lessons) and Taiwan (for the final assignment).  

##  **Project Overview**
The goal of this project was to build a model that predicts whether a company is likely to go bankrupt, using its financial metrics.  
This project required applying the **entire data science workflow** — from data ingestion to model deployment.  

Key objectives included:
- Reading and processing **compressed JSON data**.
- Handling **imbalanced datasets** (undersampling, oversampling).
- Training, tuning, and validating machine learning models (Random Forests, Gradient Boosting).
- Evaluating model performance with **confusion matrix, precision, recall, classification report**.
- Building a deployable prediction function to generate insights from new data.

---

##  **Key Steps**
### 1. Data Wrangling
- Loaded compressed JSON data (`.gz`) and created a clean DataFrame with company IDs as the index.
- Checked for missing values and handled NaNs.
- Performed class balance analysis (only ~4% of companies were bankrupt).

### 2. Feature & Target Engineering
- Defined **feature matrix `X`** and **target vector `y`** using `bankrupt` column.
- Split into training & test sets (80/20 split, reproducible with `random_state=42`).

### 3. Addressing Class Imbalance
- Applied **Random Oversampling** to balance the training dataset.
- Created oversampled feature/target sets (`X_train_over`, `y_train_over`).

### 4. Model Training & Hyperparameter Tuning
- Trained **RandomForestClassifier** on oversampled data.
- Performed cross-validation and **GridSearchCV** to find optimal hyperparameters.
- Selected the best-performing model and re-trained.

### 5. Model Evaluation
- Generated **confusion matrix** and **classification report** on the test set.
- Plotted **feature importances** (Top 10).
- Achieved strong performance with improved recall for the minority class.

### 6. Deployment
- Saved model as `model-5-5.pkl`.
- Created `my_predictor_assignment.py` with:
  - `wrangle()` – to preprocess new data
  - `make_predictions()` – to load model & generate predictions for new datasets

---

##  **Visualizations**
- **Class Balance Chart** – confirmed dataset imbalance before resampling.
- **Confusion Matrix** – assessed model performance visually.
- **Feature Importance Plot** – highlighted most influential features.

---

##  **Tech Stack**
- **Python**, **Pandas**, **NumPy**
- **scikit-learn** (RandomForest, GridSearchCV, classification metrics)
- **matplotlib**, **seaborn** for visualization


---

##  **Key Learnings**
- Learned **how to handle highly imbalanced datasets** effectively for rare events.
- Improved understanding of **hyperparameter tuning** with GridSearchCV.
- Strengthened model deployment skills with a reproducible **prediction pipeline**.
- Gained practical experience with **precision/recall trade-offs** in real-world business scenarios.

---

##  **Acknowledgments**
This project is part of the **WorldQuant University – Applied Data Science Lab**.

---

##  Connect
If you have feedback or ideas on improving this model, feel free to connect with me on linkedin or open an issue in this repo!

