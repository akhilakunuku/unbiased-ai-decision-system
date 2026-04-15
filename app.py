import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UNBAIS AI DECISION CHECKER",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding: 2rem;
    }
    .bias-low {
        color: #155724;
        background-color: #d4edda;
        border-color: #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
    }
    .bias-medium {
        color: #856404;
        background-color: #fff3cd;
        border-color: #ffeeba;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
    }
    .bias-high {
        color: #721c24;
        background-color: #f8d7da;
        border-color: #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Unbiased AI Decision Checker")
st.markdown("""
Welcome to the **Unbiased AI Decision Checker**! This application helps you detect and understand potential biases in your datasets and machine learning models.
Upload your dataset, specify the sensitive features and target, and we'll analyze it to see if a simple machine learning model exhibits biased decision-making based on sensitive attributes like gender, race, or age.
""")

# 1. File Upload
st.header("Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        # 3. Data Processing (Part 1): Clean column names
        # Remove spaces and special characters from column names
        data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '', regex=True)
        
        st.subheader("Dataset Preview")
        st.markdown(f"Shape of dataset: `{data.shape[0]} rows` and `{data.shape[1]} columns`")
        st.dataframe(data.head())
        
        # 2. Column Selection
        st.header("Step 2: Select Attributes")
        st.markdown("Choose the sensitive attribute you want to check for bias against, and the target variable you want to predict.")
        
        col1, col2 = st.columns(2)
        columns = data.columns.tolist()
        
        with col1:
            sensitive_attribute = st.selectbox("Select Sensitive Attribute (e.g., gender, race, age)", columns)
        
        with col2:
            target_variable = st.selectbox("Select Target Variable (e.g., income, loan approval)", columns)
            
        if sensitive_attribute == target_variable:
            st.error("Sensitive attribute and Target variable cannot be the same! Please select different columns.")
        else:
            if st.button("Analyze Data for Bias", type="primary"):
                with st.spinner("Processing data, training model, and analyzing bias..."):
                    
                    # 3. Data Processing (Part 2): Convert categorical data & encode target
                    processed_data = data.copy()
                    
                    # Handle missing values simply by filling or dropping
                    initial_rows = len(processed_data)
                    processed_data.dropna(inplace=True)
                    if len(processed_data) < initial_rows:
                        st.info(f"Dropped {initial_rows - len(processed_data)} rows due to missing values.")
                    
                    # Encode Target Variable if it is categorical
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(processed_data[target_variable])
                    
                    # Separate features and target
                    X = processed_data.drop(columns=[target_variable])
                    
                    # Store original sensitive attribute values to group by later
                    sensitive_attribute_values = processed_data[sensitive_attribute]
                    
                    # Encode categorical variables in X using one-hot encoding
                    # Convert to categorical if object type to ensure proper encoding
                    X_encoded = pd.get_dummies(X, drop_first=True)
                    
                    # 4. Model Training
                    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
                        X_encoded, y, sensitive_attribute_values, test_size=0.3, random_state=42
                    )
                    
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    
                    # 5. Bias Detection
                    # Predict on the test set
                    predictions = model.predict(X_test)
                    
                    # Group comparison
                    results_df = pd.DataFrame({
                        'Sensitive_Attribute': sens_test,
                        'Prediction': predictions
                    })
                    
                    # Compute average prediction for each group
                    group_rates = results_df.groupby('Sensitive_Attribute')['Prediction'].mean()
                    
                    # Calculate bias as the absolute difference between groups
                    max_rate = group_rates.max()
                    min_rate = group_rates.min()
                    bias_score = max_rate - min_rate
                    
                    # 6. Visualization & 7. Bias Interpretation
                    st.divider()
                    st.header("Step 3: Results & Bias Detection")
                    
                    st.subheader("Bias Score")
                    
                    if bias_score < 0.1:
                        bias_level = "Low"
                        st.markdown(f'<div class="bias-low">🟢 Low Bias Detected: {bias_score:.3f}</div>', unsafe_allow_html=True)
                    elif bias_score < 0.25:
                        bias_level = "Medium"
                        st.markdown(f'<div class="bias-medium">🟡 Medium Bias Detected: {bias_score:.3f}</div>', unsafe_allow_html=True)
                    else:
                        bias_level = "High"
                        st.markdown(f'<div class="bias-high">🔴 High Bias Detected: {bias_score:.3f}</div>', unsafe_allow_html=True)
                        
                    st.markdown("*(The bias score represents the maximum absolute difference in positive prediction rates between different groups of the sensitive attribute. A higher score means more disparity.)*")
                    
                    st.write("")
                    col3, col4 = st.columns([1, 2])
                    
                    with col3:
                        st.subheader("Group-wise Prediction Values")
                        
                        # Formatting the dataframe for display
                        display_df = group_rates.to_frame(name="Avg. Positive Prediction").reset_index()
                        display_df.rename(columns={'Sensitive_Attribute': sensitive_attribute}, inplace=True)
                        st.dataframe(display_df, hide_index=True)
                        
                    with col4:
                        st.subheader(f"Prediction Rate by {sensitive_attribute}")
                        
                        # Bar chart comparing groups
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(group_rates.index.astype(str), group_rates.values, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'])
                        
                        ax.set_ylabel('Average Positive Prediction', fontsize=12)
                        ax.set_title(f'Comparison of Outcomes across {sensitive_attribute}', fontsize=14)
                        ax.set_ylim(0, max(1.0, max_rate + 0.2))
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        # Add values on top of bars
                        for bar in bars:
                            yval = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
                            
                        st.pyplot(fig)
                        
                    # 8. Recommendations
                    st.divider()
                    st.header("Step 4: Recommendations & Next Steps")
                    
                    if bias_level == "Low":
                        st.success("✅ **Great job!** The simple model does not exhibit significant bias with respect to the selected sensitive attribute. The prediction rates are fairly equal across groups.")
                        st.markdown("""
                        **Recommendations:**
                        - Continue testing for fairness regularly, especially when deploying models in production.
                        - Check for bias across other sensitive attributes in your dataset.
                        - Remember that this checker looks at basic statistical parity. Other definitions of fairness might still apply depending on your use case.
                        """)
                    elif bias_level == "Medium":
                        st.warning("⚠️ **Caution!** There is a moderate level of bias. The model favors certain groups over others.")
                        st.markdown("""
                        **Recommendations:**
                        - **Investigate the data:** Check if certain groups are underrepresented or if historical target labels contain bias.
                        - **Analyze feature correlations:** See if the model relies on proxies for the sensitive attribute to make decisions.
                        - **Consider Data Balancing:** Try resampling techniques to ensure equal representation in training data.
                        """)
                    else:
                        st.error("🚨 **High Bias Detected!** The model's decisions vary significantly across demographic groups, presenting a risk of unfairness.")
                        st.markdown("""
                        **Actionable Recommendations:**
                        1. **Data Balancing:** Your dataset may be heavily imbalanced. Try oversampling underrepresented groups or aggressively collecting more representative data.
                        2. **Removing Sensitive Features (and Proxies):** Ensure you aren't training the model with the sensitive attribute if it causes direct bias. Also, look for hidden proxies (e.g., zip code might be a proxy for race/income).
                        3. **Fairness-Aware Models:** Standard logistic regression does not guarantee fairness. Implement specific fairness interventions:
                            - *Pre-processing:* Reweighing the samples in the dataset so groups have equal influence.
                            - *In-processing:* Adding fairness penalty constraints during model training.
                            - *Post-processing:* Adjusting prediction thresholds differently for each group to achieve equality of opportunity.
                        4. **Re-evaluate the Problem:** Has historical systemic bias affected the "ground truth" labels you are trying to predict? If your labels are biased, your model will be too.
                        """)

    except Exception as e:
        st.error(f"An error occurred while analyzing the dataset: {str(e)}")
        st.info("Please make sure your dataset is clean and properly formatted.")
else:
    # Display a placeholder when no file is uploaded
    st.info("👆 Please upload a CSV file to begin the analysis.")
    
    # Show example of what the tool does
    with st.expander("How does this tool work?"):
        st.markdown("""
        1. **Upload Data:** Provide your tabular dataset in CSV format.
        2. **Select Columns:** Tell the tool which column contains grouping data (e.g., Gender, Race) and which contains the label you want to predict (e.g., Approved, Defaulted).
        3. **Processing:** The app cleans the column names, transforms categorical data to numbers, and splits it into training and testing sets.
        4. **Modeling:** A Logistic Regression model is trained to predict the target variable based on the provided features.
        5. **Bias Checking:** We analyze the model's predictions on the test set, specifically looking at how the positive prediction rate differs between the groups defined by your sensitive attribute.
        """)

