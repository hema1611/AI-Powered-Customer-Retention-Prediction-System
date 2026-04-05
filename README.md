# AI-Powered-Customer-Retention-Prediction-System
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Customer Retention Prediction System | Documentation</title>
    <!-- No external CSS, clean semantic HTML for README compatibility -->
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; max-width: 1200px; margin: 0 auto; padding: 20px; color: #1a1a1a; background-color: #ffffff;">

    <!-- Title & Header -->
<h1 style="border-bottom: 3px solid #2c3e50; padding-bottom: 10px;">📊 AI-Powered Customer Retention Prediction System</h1>
    <p><strong>Prepared by:</strong> THE SKILL UNION Data Science & AI Division<br>
    <strong>Project Title:</strong> Customer Churn Prediction Using Machine Learning<br>
    <strong>Deployed Application:</strong> <a href="https://ai-powered-customer-retention-prediction-h5av.onrender.com/" target="_blank" style="color:#2563eb;">https://ai-powered-customer-retention-prediction-h5av.onrender.com/</a></p>
    
<p><strong>Contact for Support:</strong><br>
    📧 <a href="mailto:hemapalthapanchala31@gmail.com">hemapalthapanchala31@gmail.com</a><br>
    🔗 <a href="https://www.linkedin.com/in/hemalathapanchala/" target="_blank" style="color:#2563eb;">linkedin.com/in/hemalathapanchala</a></p>

    <!-- Table of Contents -->
<h2>📑 Table of Contents</h2>
<ul>
    <li><a href="#abstract" style="text-decoration: none;">1. Abstract</a></li>
    <li><a href="#acknowledgments" style="text-decoration: none;">2. Acknowledgments</a></li>
    <li><a href="#introduction" style="text-decoration: none;">3. Introduction &amp; Business Problem</a></li>
    <li><a href="#dataset" style="text-decoration: none;">4. Dataset Overview</a></li>
    <li><a href="#ml-pipeline" style="text-decoration: none;">5. Complete ML Pipeline Implementation</a></li>
    <li><a href="#webapp" style="text-decoration: none;">6. Web Application Development (Flask)</a></li>
    <li><a href="#deployment" style="text-decoration: none;">7. Cloud Deployment (Render)</a></li>
    <li><a href="#results" style="text-decoration: none;">8. Results &amp; Model Performance</a></li>
    <li><a href="#conclusion" style="text-decoration: none;">9. Conclusion &amp; Future Work</a></li>
    <li><a href="#references" style="text-decoration: none;">10. References</a></li>
</ul>

<!-- 1. Abstract -->
<h2 id="abstract">📌 1. Abstract</h2>
<p>Customer churn is a critical challenge for subscription-based businesses, leading to significant revenue loss and increased acquisition costs. Research indicates that acquiring a new customer costs <strong>five times more</strong> than retaining an existing one. This project presents a comprehensive machine-learning solution to predict customer churn proactively.</p>
<p>Utilizing the Telco Customer Churn dataset from Kaggle (7,043 customers, 21 features), we developed and deployed an end-to-end pipeline. The process includes data ingestion, rigorous preprocessing (handling 11 missing values, feature engineering with a new <code>network_provider</code> column, outlier treatment, and encoding), data balancing using SMOTE (which increased our training data from 5,634 to 8,276 balanced samples), and training multiple classification models including KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and SVM.</p>
<p>The <strong>XGBoost</strong> classifier emerged as the best-performing model, achieving a <strong>test accuracy of 77%</strong>. The final model is deployed as a user-friendly web application using Flask on the Render cloud platform, allowing stakeholders to input customer details and receive an instant churn risk assessment.</p>

<!-- 2. Acknowledgments -->
<h2 id="acknowledgments">🙏 2. Acknowledgments</h2>
<p>We would like to express our sincere gratitude to:</p>
<ul>
    <li><strong>THE SKILL UNION</strong> for providing the platform, guidance, and industry exposure.</li>
    <li>The open-source community for libraries like Scikit-learn, Pandas, XGBoost, Imbalanced-learn, and Flask.</li>
    <li>The creators of the Telco Customer Churn dataset on Kaggle (platform: blastchar).</li>
    <li>All team members and mentors who contributed their time and expertise.</li>
</ul>

<!-- 3. Introduction & Business Problem -->
<h2 id="introduction">🏢 3. Introduction &amp; Business Problem</h2>
<h3>3.1 What is Customer Churn?</h3>
<p><strong>Customer Churn</strong> refers to the percentage of customers who stop using a company's product or service over a specific period. In telecommunications, churn occurs when customers switch to a competitor or cancel subscriptions.</p>
<h3>3.2 Why is Churn Prediction Important?</h3>
<table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
    <thead style="background-color: #f0f0f0;"><tr><th>Challenge</th><th>Impact</th></tr></thead>
    <tbody>
        <tr><td>High acquisition cost</td><td>Acquiring new customers costs 5x more than retaining existing ones</td></tr>
        <tr><td>Revenue leakage</td><td>Uncontrolled churn reduces monthly recurring revenue (MRR)</td></tr>
        <tr><td>Brand reputation</td><td>High churn indicates poor service quality</td></tr>
        <tr><td>Competitive disadvantage</td><td>Competitors gain market share with every lost customer</td></tr>
    </tbody>
</table>
<h3>3.3 Project Objectives</h3>
<ol>
    <li><strong>Predict:</strong> Identify customers with high churn probability.</li>
    <li><strong>Understand:</strong> Discover key drivers behind churn.</li>
    <li><strong>Act:</strong> Provide actionable insights for retention teams.</li>
    <li><strong>Deploy:</strong> Create a real-time prediction tool.</li>
</ol>

<!-- 4. Dataset Overview -->
<h2 id="dataset">🗃️ 4. Dataset Overview</h2>
<p><strong>Source:</strong> Kaggle · <strong>File:</strong> WA_Fn-UseC_-Telco-Customer-Churn.csv · <strong>Size:</strong> 7,043 rows × 21 columns</p>
<p><strong>Key Features:</strong> customerID, gender, SeniorCitizen, tenure, PhoneService, InternetService, Contract, MonthlyCharges, TotalCharges, Churn (target).</p>
<p><strong>Initial Inspection:</strong> No null values detected initially, but <code>TotalCharges</code> had 11 missing values after type conversion (empty strings).</p>

<!-- 5. ML Pipeline -->
<h2 id="ml-pipeline">⚙️ 5. Complete ML Pipeline Implementation (Step-by-Step)</h2>
<h3>5.1 Data Ingestion &amp; Feature Engineering</h3>
<p>Added <code>network_provider</code> column mapped from PaymentMethod (Airtel, Jio, Vi, BSNL). Converted TotalCharges to numeric → 11 nulls discovered.</p>
<h3>5.2 Train-Test Split &amp; Target Encoding</h3>
<p><strong>Split:</strong> 80/20 ratio → X_train: 5,634, X_test: 1,409. Target mapped: 'Yes'→1 (churn), 'No'→0.</p>
<h3>5.3 Handling Missing Values (Mode Imputation)</h3>
<p>Compared 10+ imputation techniques, mode imputation preserved distribution best. Filled 11 missing values in TotalCharges with mode.</p>
<h3>5.4 Numerical vs. Categorical Split</h3>
<p>Numerical columns: SeniorCitizen, tenure, MonthlyCharges, TotalCharges_mode. Categorical: 17 features (including customerID later dropped).</p>
<h3>5.5 Transformation &amp; Outlier Treatment (Quantile + Trimming)</h3>
<p>QuantileTransformer (normal distribution) + IQR trimming applied to numerical features → new columns with '_trim' suffix.</p>
<h3>5.6 Feature Selection (Filter Method)</h3>
<p>VarianceThreshold (threshold=0.01) removed SeniorCitizen_trim. Pearson correlation p-value &lt; 0.05 confirmed 3 significant features: tenure_trim, MonthlyCharges_trim, TotalCharges_mode_trim.</p>
<h3>5.7 Categorical Encoding (OneHot + Ordinal)</h3>
<p>OneHotEncoding for nominal (gender, Partner, etc.) → 28 columns. OrdinalEncoding for Contract (Month-to-month=0, One year=1, Two year=2). Final combined training data shape: (5634, 31).</p>
<h3>5.8 Data Balancing (SMOTE)</h3>
<p><strong>Before:</strong> No Churn 4,138, Churn 1,496 (73.4% / 26.6%). <strong>After SMOTE:</strong> 4,138 each → perfectly balanced (8276 samples).</p>
<h3>5.9 Feature Scaling (StandardScaler)</h3>
<p>StandardScaler (mean=0, std=1) applied to numerical features. Saved scaler for deployment.</p>
<h3>5.10 Model Training &amp; Evaluation</h3>
<p>Evaluated 8 models: KNN, LR, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, SVM. <strong>XGBoost outperformed all</strong> with 77% test accuracy, 0.78 ROC-AUC.</p>
<h3>5.11 Model Saving</h3>
<p>Pickled model and scaler as <code>Model.pkl</code> and <code>standar_scaler.pkl</code>.</p>

<!-- 6. Web Application Development -->
<h2 id="webapp">🌐 6. Web Application Development (Flask)</h2>
<p>The Flask app replicates all preprocessing steps (31 features) and serves an interactive HTML form. User inputs customer details → backend encodes features → StandardScaler transforms → XGBoost predicts → returns "Churn" or "No Churn".</p>
<p><strong>Key encoding examples:</strong> gender_Male (1/0), MultipleLines (two columns), InternetService (two columns), Contract ordinal, PaymentMethod (one-hot), network_provider (one-hot), service add-ons (two columns each).</p>
<p><strong>Frontend features:</strong> Network provider cards, dark/light/slate theme switcher, organized form sections, animated result card.</p>

<!-- 7. Cloud Deployment -->
<h2 id="deployment">☁️ 7. Cloud Deployment (Render)</h2>
<p><strong>Steps:</strong> GitHub repo → requirements.txt (flask, xgboost, scikit-learn, numpy) → Render web service connected → build command: <code>pip install -r requirements.txt</code> → start: <code>gunicorn app:app</code>.</p>
<p><strong>Live URL:</strong> <a href="https://ai-powered-customer-retention-prediction-h5av.onrender.com/" target="_blank">https://ai-powered-customer-retention-prediction-h5av.onrender.com/</a></p>

<!-- 8. Results & Performance -->
<h2 id="results">📈 8. Results &amp; Model Performance</h2>
<p><strong>Best Model: XGBoost</strong></p>
<table border="1" cellpadding="8" style="border-collapse: collapse;">
    <tr><th>Metric</th><th>Score</th></tr>
    <tr><td>Test Accuracy</td><td>77%</td></tr>
    <tr><td>Precision (Churn)</td><td>0.58</td></tr>
    <tr><td>Recall (Churn)</td><td>0.73</td></tr>
    <tr><td>F1-Score (Churn)</td><td>0.65</td></tr>
    <tr><td>ROC-AUC</td><td>0.78</td></tr>
</table>
<p><strong>Confusion Matrix:</strong></p>
<pre style="background:#f6f8fa; padding:12px; border-radius:6px;">
          Predicted
          No Churn  Churn
Actual No Churn   789     212
Actual Churn      111     297</pre>
<p><strong>Business Impact:</strong> Correctly identified 297 churners out of 408 (73% recall), enabling proactive retention.</p>

<!-- 9. Conclusion & Future Work -->
<h2 id="conclusion">🔮 9. Conclusion &amp; Future Work</h2>
<p>This project successfully built and deployed an end-to-end churn prediction system. XGBoost achieved 77% accuracy with robust preprocessing and SMOTE balancing. The live web app empowers business teams to take data-driven actions.</p>
<h3>Future Enhancements</h3>
<ul>
    <li>Sentiment analysis from support calls</li>
    <li>Real-time CRM integration</li>
    <li>Deep learning (LSTM) for sequential patterns</li>
    <li>Automated retention campaign triggers</li>
    <li>SHAP explanations for model interpretability</li>
</ul>

<!-- 10. References -->
<h2 id="references">📚 10. References</h2>
<ul>
    <li>Telco Customer Churn Dataset, Kaggle: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">https://www.kaggle.com/datasets/blastchar/telco-customer-churn</a></li>
    <li>Scikit-learn, XGBoost, Flask, Imbalanced-learn documentation</li>
    <li>Render.com Web Services deployment</li>
</ul>

<hr style="margin: 30px 0;">
<p style="text-align: center;"><strong>Document Prepared By:</strong> Hemalatha Panchala | <strong>Date:</strong> April 2026 | <strong>Version:</strong> 1.0</p>
<p style="text-align: center; font-style: italic;">“Predict churn before it happens. Retain customers intelligently.”</p>
</body>
</html>
