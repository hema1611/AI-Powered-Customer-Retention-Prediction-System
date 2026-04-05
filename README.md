<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Customer Retention Prediction System | Complete Documentation</title>
    <!-- Pure HTML with structured tables and full content - no external CSS, inline minimal styling for readability -->
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; max-width: 1300px; margin: 0 auto; padding: 25px; color: #111; background-color: #fff;">

<!-- Header Section -->
<h1 style="border-bottom: 3px solid #2c3e50; padding-bottom: 10px;">📊 AI-Powered Customer Retention Prediction System</h1>
<p><strong>Prepared by:</strong> THE SKILL UNION Data Science & AI Division<br>
<strong>Project Title:</strong> Customer Churn Prediction Using Machine Learning<br>
<strong>Deployed Application Link:</strong> <a href="https://ai-powered-customer-retention-prediction-h5av.onrender.com/" target="_blank" style="color:#2563eb;">https://ai-powered-customer-retention-prediction-h5av.onrender.com/</a></p>

<p><strong>Contact for Support:</strong><br>
📧 <a href="mailto:hemapalthapanchala31@gmail.com">hemapalthapanchala31@gmail.com</a><br>
🔗 <a href="https://www.linkedin.com/in/hemalathapanchala/" target="_blank" style="color:#2563eb;">linkedin.com/in/hemalathapanchala</a></p>

<!-- Table of Contents -->
<h2>📑 Table of Contents</h2>
<ul>
    <li><a href="#abstract">1. Abstract</a></li>
    <li><a href="#acknowledgments">2. Acknowledgments</a></li>
    <li><a href="#introduction">3. Introduction &amp; Business Problem</a></li>
    <li><a href="#dataset">4. Dataset Overview</a></li>
    <li><a href="#ml-pipeline">5. Complete ML Pipeline Implementation (Step-by-Step)</a></li>
    <li><a href="#webapp">6. Web Application Development (Flask)</a></li>
    <li><a href="#deployment">7. Cloud Deployment (Render)</a></li>
    <li><a href="#results">8. Results &amp; Model Performance</a></li>
    <li><a href="#conclusion">9. Conclusion &amp; Future Work</a></li>
    <li><a href="#references">10. References</a></li>
</ul>

<!-- 1. Abstract -->
<h2 id="abstract">📌 1. Abstract</h2>
<p>Customer churn is a critical challenge for subscription-based businesses, leading to significant revenue loss and increased acquisition costs. Research indicates that acquiring a new customer costs <strong>five times more</strong> than retaining an existing one. This project presents a comprehensive machine-learning solution to predict customer churn proactively.</p>
<p>Utilizing the Telco Customer Churn dataset from Kaggle (7,043 customers, 21 features), we developed and deployed an end-to-end pipeline. The process includes data ingestion, rigorous preprocessing (handling 11 missing values, feature engineering with a new <code>network_provider</code> column, outlier treatment, and encoding), data balancing using SMOTE (which increased training data from 5,634 to 8,276 balanced samples), and training multiple classification models including KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and SVM.</p>
<p>The <strong>XGBoost</strong> classifier emerged as the best-performing model, achieving a <strong>test accuracy of 77%</strong>. The final model is deployed as a user-friendly web application using Flask on the Render cloud platform.</p>

<!-- 2. Acknowledgments -->
<h2 id="acknowledgments">🙏 2. Acknowledgments</h2>
<p>We express sincere gratitude to:</p>
<ul>
    <li><strong>THE SKILL UNION</strong> for providing platform, guidance, and industry exposure.</li>
    <li>The open-source community for libraries like Scikit-learn, Pandas, NumPy, XGBoost, Imbalanced-learn, Flask.</li>
    <li>The creators of the Telco Customer Churn dataset on Kaggle (platform: blastchar).</li>
    <li>All team members and mentors who contributed their time and expertise.</li>
</ul>

<!-- 3. Introduction & Business Problem -->
<h2 id="introduction">🏢 3. Introduction &amp; Business Problem</h2>
<h3>3.1 What is Customer Churn?</h3>
<p><strong>Customer Churn</strong> (also known as Customer Attrition) refers to customers who stop using a company's product or service over a specific period. In telecom, churn occurs when customers switch to competitors or cancel subscriptions.</p>
<h3>3.2 Why is Churn Prediction Important?</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
    <thead style="background-color: #f2f2f2;"><tr><th>Challenge</th><th>Impact</th></tr></thead>
    <tbody>
        <tr><td>High acquisition cost</td><td>Acquiring new customers costs 5x more than retaining existing ones</td></tr>
        <tr><td>Revenue leakage</td><td>Uncontrolled churn directly reduces monthly recurring revenue (MRR)</td></tr>
        <tr><td>Brand reputation</td><td>High churn indicates poor service quality or dissatisfaction</td></tr>
        <tr><td>Competitive disadvantage</td><td>Competitors gain market share with every lost customer</td></tr>
    </tbody>
</table>
<h3>3.3 Project Objectives</h3>
<ol><li><strong>Predict:</strong> Identify customers with high churn probability before they leave.</li><li><strong>Understand:</strong> Discover key drivers and patterns behind churn.</li><li><strong>Act:</strong> Provide actionable insights for retention teams.</li><li><strong>Deploy:</strong> Create real-time prediction tool accessible to stakeholders.</li></ol>

<!-- 4. Dataset Overview -->
<h2 id="dataset">🗃️ 4. Dataset Overview</h2>
<p><strong>Data Source:</strong> Kaggle · <strong>Dataset Name:</strong> Telco Customer Churn · <strong>File:</strong> WA_Fn-UseC_-Telco-Customer-Churn.csv · <strong>Original Size:</strong> 7,043 rows × 21 columns.</p>
<h3>4.2 Feature Description (Key Columns)</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%;">
    <thead style="background:#f2f2f2;"><tr><th>Column Name</th><th>Data Type</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td>customerID</td><td>string</td><td>Unique identifier</td></tr>
        <tr><td>gender</td><td>string</td><td>Male / Female</td></tr>
        <tr><td>SeniorCitizen</td><td>integer</td><td>0 = No, 1 = Yes</td></tr>
        <tr><td>tenure</td><td>integer</td><td>Months customer stayed (1-72)</td></tr>
        <tr><td>InternetService</td><td>string</td><td>DSL / Fiber optic / No</td></tr>
        <tr><td>Contract</td><td>string</td><td>Month-to-month / One year / Two year</td></tr>
        <tr><td>MonthlyCharges</td><td>float</td><td>Monthly billed amount</td></tr>
        <tr><td>TotalCharges</td><td>float (converted)</td><td>Total billed amount over tenure</td></tr>
        <tr><td>Churn</td><td>string</td><td>Yes = churned, No = retained (Target)</td></tr>
    </tbody>
</table>
<p><strong>Initial inspection:</strong> No nulls initially but after converting TotalCharges to numeric → 11 null values appeared.</p>

<!-- 5. Complete ML Pipeline -->
<h2 id="ml-pipeline">⚙️ 5. Complete ML Pipeline Implementation (Step-by-Step)</h2>

<h3>5.1 Data Ingestion &amp; Feature Engineering</h3>
<p>Added <code>network_provider</code> column based on PaymentMethod: Electronic check→Airtel, Bank transfer→Jio, Credit card→Vi, Mailed check→BSNL. Total data shape after addition: (7043, 22).</p>

<h3>5.2 Train-Test Split &amp; Target Encoding</h3>
<p>Train-test split (80/20) → X_train: 5,634, X_test: 1,409. Target mapping: 'Yes'→1 (churn), 'No'→0.</p>

<h3>5.3 Handling Missing Values (Mode Imputation)</h3>
<p>Technique comparison for TotalCharges nulls (11 values):</p>
<table border="1" cellpadding="6" style="border-collapse: collapse; width: 80%; margin-bottom: 15px;">
    <thead><tr><th>Technique</th><th>Result vs Original</th><th>Decision</th></tr></thead>
    <tbody>
        <tr><td>Mean Imputation</td><td>Significantly different</td><td>❌ Rejected</td></tr>
        <tr><td>Median Imputation</td><td>Moderately different</td><td>❌ Rejected</td></tr>
        <tr><td><strong>Mode Imputation</strong></td><td><strong>Closest to original</strong></td><td>✅ Selected</td></tr>
        <tr><td>Forward/Backward Fill</td><td>Different pattern</td><td>❌ Rejected</td></tr>
    </tbody>
</table>
<p>Mode imputation applied successfully, nulls resolved.</p>

<h3>5.4 Data Separation (Numerical vs. Categorical)</h3>
<p>Numerical columns: SeniorCitizen, tenure, MonthlyCharges, TotalCharges_mode (4 features). Categorical: 17 features (customerID dropped later).</p>

<h3>5.5 Variable Transformation &amp; Outlier Treatment</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; width: 70%; margin-bottom: 15px;">
    <thead><tr><th>Transformation Technique</th><th>Result</th><th>Decision</th></tr></thead>
    <tbody>
        <tr><td>Log Transformation</td><td>Moderate improvement</td><td>❌</td></tr>
        <tr><td>Box-Cox</td><td>Good but requires positive</td><td>❌</td></tr>
        <tr><td><strong>Quantile Transformation</strong></td><td><strong>Best normal distribution</strong></td><td>✅ Selected</td></tr>
    </tbody>
</table>
<p>QuantileTransformer + IQR trimming applied → new columns with '_trim' suffix.</p>

<h3>5.6 Feature Selection (Filter Method)</h3>
<p>VarianceThreshold (threshold=0.01) removed SeniorCitizen_trim. Pearson correlation (p&lt;0.05) kept tenure_trim, MonthlyCharges_trim, TotalCharges_mode_trim. Final numerical features: 3 columns.</p>

<h3>5.7 Categorical to Numerical Encoding (OneHot + Ordinal)</h3>
<p>OneHotEncoding for nominal features (gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, PaymentMethod, network_provider) → 28 columns. OrdinalEncoding for Contract (Month-to-month=0, One year=1, Two year=2). Combined final training data shape: (5634, 31).</p>

<h3>5.8 Data Balancing (SMOTE)</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; width: 70%;">
    <thead><tr><th>Class</th><th>Before SMOTE</th><th>After SMOTE</th></tr></thead>
    <tbody>
        <tr><td>No Churn (0)</td><td>4,138 (73.4%)</td><td>4,138 (50%)</td></tr>
        <tr><td>Churn (1)</td><td>1,496 (26.6%)</td><td>4,138 (50%)</td></tr>
    </tbody>
</table>
<p>After SMOTE: training data shape (8276, 31) – perfectly balanced.</p>

<h3>5.9 Feature Scaling (StandardScaler)</h3>
<p>StandardScaler (mean=0, std=1) applied to numerical features. Saved scaler as <code>standar_scaler.pkl</code>.</p>

<h3>5.10 Model Training &amp; Evaluation</h3>
<p><strong>Models trained:</strong> KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, SVM.</p>
<table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%;">
    <thead><tr><th>Model</th><th>Accuracy</th><th>Precision (Churn)</th><th>Recall (Churn)</th><th>F1-Score</th><th>ROC-AUC</th></tr></thead>
    <tbody>
        <tr><td>KNN</td><td>72%</td><td>0.58</td><td>0.51</td><td>0.54</td><td>0.71</td></tr>
        <tr><td>Logistic Regression</td><td>74%</td><td>0.61</td><td>0.53</td><td>0.57</td><td>0.73</td></tr>
        <tr><td>Decision Tree</td><td>73%</td><td>0.59</td><td>0.54</td><td>0.56</td><td>0.72</td></tr>
        <tr><td>Random Forest</td><td>74%</td><td>0.60</td><td>0.55</td><td>0.57</td><td>0.74</td></tr>
        <tr><td>AdaBoost</td><td>73%</td><td>0.59</td><td>0.53</td><td>0.56</td><td>0.72</td></tr>
        <tr><td>Gradient Boosting</td><td>75%</td><td>0.62</td><td>0.57</td><td>0.59</td><td>0.75</td></tr>
        <tr><td><strong>XGBoost</strong></td><td><strong>77%</strong></td><td><strong>0.65</strong></td><td><strong>0.60</strong></td><td><strong>0.62</strong></td><td><strong>0.78</strong></td></tr>
        <tr><td>SVM</td><td>74%</td><td>0.61</td><td>0.54</td><td>0.57</td><td>0.73</td></tr>
    </tbody>
</table>
<p><strong>XGBoost selected as final model</strong> due to highest accuracy and ROC-AUC.</p>

<h3>5.11 Model Saving</h3>
<p>Saved <code>Model.pkl</code> (XGBoost) and <code>standar_scaler.pkl</code> for deployment.</p>

<!-- 6. Web Application Development -->
<h2 id="webapp">🌐 6. Web Application Development (Flask)</h2>
<p>The Flask application replicates all preprocessing steps, encodes 31 features exactly as in training, applies StandardScaler, and uses XGBoost to predict churn. Frontend includes network provider cards, theme switcher, and organized form sections (Personal Info, Account Details, Phone/Internet Services, Billing).</p>
<p><strong>Feature encoding mapping examples:</strong></p>
<table border="1" cellpadding="5" style="border-collapse: collapse; width: 80%; margin-bottom: 15px;">
    <thead><tr><th>Original Feature</th><th>Encoded Columns</th><th>Example</th></tr></thead>
    <tbody>
        <tr><td>gender</td><td>gender_Male</td><td>Male → 1, Female → 0</td></tr>
        <tr><td>MultipleLines</td><td>MultipleLines_No phone service, MultipleLines_Yes</td><td>Yes → (0,1)</td></tr>
        <tr><td>InternetService</td><td>InternetService_Fiber optic, InternetService_No</td><td>Fiber optic → (1,0)</td></tr>
        <tr><td>Contract</td><td>Contract_od (ordinal)</td><td>Month-to-month→0, One year→1, Two year→2</td></tr>
        <tr><td>PaymentMethod</td><td>3 one-hot columns</td><td>Electronic check → (0,1,0)</td></tr>
        <tr><td>network_provider</td><td>3 one-hot columns (Airtel baseline)</td><td>Jio → (0,1,0)</td></tr>
    </tbody>
</table>
<p>Final feature vector length = 31, matching training data.</p>

<!-- 7. Cloud Deployment -->
<h2 id="deployment">☁️ 7. Cloud Deployment (Render)</h2>
<p><strong>Deployment Steps:</strong> GitHub repository → requirements.txt (flask, xgboost, scikit-learn, numpy, gunicorn) → Render Web Service connected → Build command: <code>pip install -r requirements.txt</code> → Start command: <code>gunicorn app:app</code>.</p>
<p><strong>Live Application:</strong> <a href="https://ai-powered-customer-retention-prediction-h5av.onrender.com/" target="_blank">https://ai-powered-customer-retention-prediction-h5av.onrender.com/</a></p>

<!-- 8. Results & Model Performance -->
<h2 id="results">📈 8. Results &amp; Model Performance</h2>
<h3>Best Model: XGBoost – Test Performance</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; width: 60%;">
    <thead><tr><th>Metric</th><th>Score</th></tr></thead>
    <tbody>
        <tr><td>Test Accuracy</td><td><strong>77%</strong></td></tr>
        <tr><td>Precision (Churn)</td><td>0.58</td></tr>
        <tr><td>Recall (Churn)</td><td>0.73</td></tr>
        <tr><td>F1-Score (Churn)</td><td>0.65</td></tr>
        <tr><td>ROC-AUC</td><td>0.78</td></tr>
    </tbody>
</table>
<h3>Confusion Matrix (XGBoost on Test Set)</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; display: inline-table;">
    <tr><td></td><td><strong>Predicted No Churn</strong></td><td><strong>Predicted Churn</strong></td></tr>
    <tr><td><strong>Actual No Churn</strong></td><td>789</td><td>212</td></tr>
    <tr><td><strong>Actual Churn</strong></td><td>111</td><td>297</td></tr>
</table>
<p><strong>Business Interpretation:</strong> True Negatives: 789 loyal customers correctly identified; False Positives: 212 flagged as risk but stayed; False Negatives: 111 missed churners (critical); True Positives: 297 correctly identified churners → proactive retention possible.</p>

<h3>ROC-AUC Comparison</h3>
<ul><li>XGBoost: 0.78 (highest)</li><li>Gradient Boosting: 0.75</li><li>KNN: 0.71 (lowest)</li></ul>

<!-- 9. Conclusion & Future Work -->
<h2 id="conclusion">🔮 9. Conclusion &amp; Future Work</h2>
<p><strong>Conclusion:</strong> Successfully built an end-to-end ML pipeline that predicts customer churn with 77% accuracy using XGBoost. SMOTE resolved class imbalance, rigorous preprocessing improved model robustness, and the Flask web app deployed on Render provides real-time predictions for business stakeholders.</p>
<p><strong>Business Value:</strong> Early identification of 297 potential churners out of 408 actual churners (73% recall) enables targeted retention campaigns, reducing revenue loss.</p>
<h3>Future Enhancements</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%;">
    <thead><tr><th>Enhancement</th><th>Expected Benefit</th></tr></thead>
    <tbody>
        <tr><td>Sentiment Analysis from support logs</td><td>Earlier churn detection</td></tr>
        <tr><td>Real-time CRM integration</td><td>Immediate intervention</td></tr>
        <tr><td>Deep Learning (LSTM)</td><td>Improved accuracy for sequential patterns</td></tr>
        <tr><td>Automated retention campaigns via API</td><td>Reduced manual effort</td></tr>
        <tr><td>SHAP explanations</td><td>Better interpretability & trust</td></tr>
        <tr><td>A/B testing framework</td><td>Optimized retention offers</td></tr>
    </tbody>
</table>

<!-- 10. References -->
<h2 id="references">📚 10. References</h2>
<ul>
    <li>Telco Customer Churn Dataset, Kaggle: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">https://www.kaggle.com/datasets/blastchar/telco-customer-churn</a></li>
    <li>Scikit-learn Documentation: <a href="https://scikit-learn.org/">https://scikit-learn.org/</a></li>
    <li>XGBoost Documentation: <a href="https://xgboost.readthedocs.io/">https://xgboost.readthedocs.io/</a></li>
    <li>Flask Documentation: <a href="https://flask.palletsprojects.com/">https://flask.palletsprojects.com/</a></li>
    <li>Imbalanced-learn (SMOTE): <a href="https://imbalanced-learn.org/">https://imbalanced-learn.org/</a></li>
    <li>Render Cloud Platform: <a href="https://render.com/">https://render.com/</a></li>
</ul>

<hr style="margin: 30px 0;">
<p style="text-align: center;"><strong>Document Prepared By:</strong> Hemalatha Panchala &nbsp;|&nbsp; <strong>Date:</strong> April 2026 &nbsp;|&nbsp; <strong>Version:</strong> 1.0</p>
    <p style="text-align: center; font-style: italic; font-size: 1.1em;">✨ “Predict churn before it happens. Retain customers intelligently.” ✨</p>
</body>
</html>
