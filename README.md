## VM Failure Prediction Analysis

This project explores two distinct approaches to predict Virtual Machine (VM) failures using telemetry data: **Survival Analysis** and a traditional **Machine Learning Classifier**. 
The goal is to not only predict failures but also to understand the factors contributing to them through interpretable methods.

***

### 1. Project Goal

The primary objective is to analyze a dataset of VM telemetry and develop models that can:
1.  Predict the **time-to-failure** for VMs (Survival Analysis).
2.  Predict the **occurrence of a failure** (Classification).
3.  Provide a clear **explanation** for why a particular prediction was made.

***

### 2. Methodology

The project follows a multi-stage process, combining data engineering, predictive modeling, and model interpretation.

#### Data Preparation
* **Feature Engineering:** New features such as `cpu_mem_ratio` and `cpu_disk_interaction` were created to capture complex system behaviors.
* **Multicollinearity Handling:** A correlation matrix was used to identify and manage highly correlated features to prevent model instability. For example, the highly correlated `disk_io` and `cpu_disk_interaction` features were handled by selecting one over the other for different models.
* **Outlier and Scaling:** Features were clipped to handle extreme outliers and then standardized using `StandardScaler` to ensure all variables were on a similar scale, which is crucial for model convergence and performance.

#### Survival Analysis
* The `lifelines` library was used to build a **Cox Time-Varying Proportional Hazards Model**. This model is designed to analyze event data and determine how various features impact the *hazard rate*, or the probability of an event (failure) occurring at a specific time.
* **Key Finding:** The survival analysis revealed that the selected features did not have a statistically significant relationship with the timing of VM failures. This suggests that the current set of features is not sufficient for predicting *when* a failure will occur.

#### Machine Learning Classification
* An **XGBoost Classifier** was trained to predict whether a failure would occur. This model is well-suited for tabular data and known for its high predictive performance.
* The model was validated using a **time series cross-validation** approach, which respects the temporal order of the data and provides a more realistic measure of performance.

#### Explainable AI (SHAP)
* **SHAP (SHapley Additive exPlanations)** values were computed to interpret the XGBoost model's predictions. The SHAP waterfall plot visually breaks down how each feature contributes to a single prediction. For a given VM, this allows us to see which features (e.g., high `net_latency`) pushed the model to predict "no failure," while others (e.g., high `cpu_util`) pushed it toward "failure."

***

### 3. Key Findings

* The features used are **not strong predictors of the *timing* of a VM failure**, according to the survival analysis.
* The XGBoost classifier was able to achieve **good predictive accuracy** on the task of classifying whether a failure will occur.
* **SHAP analysis provides critical interpretability**, allowing us to understand the "why" behind individual predictions. This insight is essential for building trust in the model and for taking targeted action to prevent potential failures.
