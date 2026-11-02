import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Guest Satisfaction Prediction", layout="wide")
st.title("Airbnb Guest Satisfaction Prediction")
st.markdown("""
Use machine learning to predict guest review scores based on Airbnb listing features.
This app guides you through data cleaning, preprocessing, feature engineering, model training, and evaluation.
""")

# ------------------------------------------------------------------
# Dataset upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your Airbnb dataset (CSV format)", type=["csv"])
use_sample = st.checkbox("No dataset? Use the built-in sample data", value=False)

# FIX 1: stop the script until data is available, so nothing below
# runs (and crashes) when `df` does not exist yet.
if uploaded_file is None and not use_sample:
    st.info("Upload a CSV file, or tick 'Use the built-in sample data' to try the app.")
    st.stop()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
else:
    sample_path = Path(__file__).parent / "sample_data.csv"
    if not sample_path.exists():
        st.error("sample_data.csv was not found in the repository.")
        st.stop()
    df = pd.read_csv(sample_path, low_memory=False)
    st.caption("Using the built-in sample dataset.")
st.subheader("Raw Dataset Sample Preview")
st.write(df.head())
st.write(f"Shape: {df.shape}")

if 'review_scores_rating' not in df.columns:
    st.error("The dataset must contain a 'review_scores_rating' column.")
    st.stop()

# ------------------------------------------------------------------
# Data cleaning
# ------------------------------------------------------------------
st.subheader("Data Cleaning and Preprocessing")
st.markdown("Cleaning involves handling missing values, converting data types, and encoding categorical variables.")

columns_to_drop = [
    'listing_url', 'summary', 'space', 'description', 'neighborhood_overview',
    'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url',
    'host_url', 'host_about', 'host_location', 'host_neighbourhood', 'street',
    'neighbourhood', 'market', 'smart_location', 'country_code', 'latitude',
    'longitude', 'first_review', 'last_review', 'host_name', 'city', 'state'
]

price_columns = ['nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee', 'extra_people']
for col in price_columns:
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

bool_cols = [
    'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'is_location_exact', 'instant_bookable', 'is_business_travel_ready',
    'requires_license', 'require_guest_profile_picture', 'require_guest_phone_verification'
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'t': True, 'f': False})

if 'host_since' in df.columns:
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['years_as_host'] = (pd.to_datetime('2025-04-25') - df['host_since']).dt.days / 365.25
    df.drop(columns=['host_since'], inplace=True)

if 'host_response_rate' in df.columns and df['host_response_rate'].dtype == object:
    df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float)

df.drop_duplicates(inplace=True)
df.dropna(thresh=int(0.7 * len(df.columns)), inplace=True)

# FIX 6: avoid the deprecated chained `fillna(..., inplace=True)` pattern
df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())

# Drop text / irrelevant columns now, before feature engineering
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

st.write("Cleaned Dataset Sample Preview")
st.write(df.head())
st.write(f"Shape after cleaning: {df.shape}")

# ------------------------------------------------------------------
# Feature engineering
# (FIX 3: done BEFORE outlier removal so engineered columns get handled too)
# ------------------------------------------------------------------
st.subheader("Feature Engineering")
st.markdown("Creating new features to capture deeper relationships in the data.")

if 'nightly_price' in df.columns and 'accommodates' in df.columns:
    df['price_per_person'] = df['nightly_price'] / df['accommodates'].replace(0, np.nan)
    df['price_per_person'] = df['price_per_person'].replace([np.inf, -np.inf], np.nan)

if 'nightly_price' in df.columns and 'bedrooms' in df.columns:
    df['price_per_bedroom'] = df['nightly_price'] / df['bedrooms'].replace(0, np.nan)
    df['price_per_bedroom'] = df['price_per_bedroom'].replace([np.inf, -np.inf], np.nan)

if all(c in df.columns for c in ['cleaning_fee', 'security_deposit', 'extra_people']):
    df['total_fees'] = (df['cleaning_fee'].fillna(0)
                        + df['security_deposit'].fillna(0)
                        + df['extra_people'].fillna(0))

if 'beds' in df.columns and 'bedrooms' in df.columns:
    df['beds_per_bedroom'] = df['beds'] / df['bedrooms'].replace(0, np.nan)
    df['beds_per_bedroom'] = df['beds_per_bedroom'].replace([np.inf, -np.inf], np.nan)

if 'security_deposit' in df.columns:
    df['has_security_deposit'] = df['security_deposit'].apply(lambda x: 1 if x > 0 else 0)

if 'cleaning_fee' in df.columns:
    df['has_cleaning_fee'] = df['cleaning_fee'].apply(lambda x: 1 if x > 0 else 0)

top_amenities = ['Wifi', 'Kitchen', 'Heating', 'Washer', 'Air conditioning', 'Free parking']

# FIX 2: every use of `amenities` is guarded, so a missing column can't raise KeyError
if 'amenities' in df.columns:
    df['amenities_list'] = df['amenities'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['amenities_count'] = df['amenities_list'].apply(len)
    df.drop(columns=['amenities_list'], inplace=True)

    for amenity in top_amenities:
        df[f'has_{amenity.lower().replace(" ", "_")}'] = df['amenities'].apply(
            lambda x, a=amenity: 1 if isinstance(x, str) and a.lower() in x.lower() else 0
        )

if 'host_is_superhost' in df.columns and 'years_as_host' in df.columns:
    df['superhost_experience'] = df['host_is_superhost'].fillna(False).astype(int) * df['years_as_host']

if 'minimum_nights' in df.columns and 'maximum_nights' in df.columns:
    df['min_max_nights_ratio'] = df['minimum_nights'] / df['maximum_nights'].replace(0, np.nan)
    df['min_max_nights_ratio'] = df['min_max_nights_ratio'].replace([np.inf, -np.inf], np.nan)

st.write("Dataset with Engineered Features")
engineered_cols = (
    ['price_per_person', 'price_per_bedroom', 'total_fees', 'beds_per_bedroom',
     'amenities_count', 'superhost_experience', 'min_max_nights_ratio']
    + [f'has_{a.lower().replace(" ", "_")}' for a in top_amenities]
    + ['has_security_deposit', 'has_cleaning_fee']
)
available_engineered_cols = [col for col in engineered_cols if col in df.columns]
st.write(df[available_engineered_cols].head())

# ------------------------------------------------------------------
# Outlier removal
# ------------------------------------------------------------------
st.subheader("Outlier Removal")


def remove_outliers_iqr(data, columns, factor=1.5):
    for col in columns:
        if col not in data.columns:
            continue
        # FIX 4: standard IQR quantiles (0.25 / 0.75)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            # Column is mostly constant (e.g. security_deposit mostly 0);
            # IQR filtering would wipe out every non-zero row, so skip it.
            continue
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        initial_rows = data.shape[0]
        # Keep NaNs here; they are imputed later in the pipeline
        keep = data[col].isna() | ((data[col] >= lower_bound) & (data[col] <= upper_bound))
        data = data[keep]
        st.write(f"Removed outliers in **{col}**: {initial_rows - data.shape[0]} rows removed.")
    return data


numerical_cols_for_outliers = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit', 'cleaning_fee',
    'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'number_of_stays',
    'host_response_rate', 'guests_included', 'price_per_person', 'price_per_bedroom',
    'total_fees', 'beds_per_bedroom', 'years_as_host', 'amenities_count'
]
numerical_cols_for_outliers = [col for col in numerical_cols_for_outliers if col in df.columns]

df = remove_outliers_iqr(df, numerical_cols_for_outliers)
st.write(f"Shape after outlier removal: {df.shape}")

if df.empty:
    st.error("All rows were removed during cleaning. Check your dataset.")
    st.stop()

# ------------------------------------------------------------------
# Skewness correction
# ------------------------------------------------------------------
st.subheader("Skewness Detection and Correction")
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
skewness = df[numeric_features].skew().sort_values(ascending=False)
skewed_cols = skewness[skewness.abs() > 0.5].index.tolist()

# FIX 5: only log-transform columns where it is valid and meaningful:
#   - not the target
#   - no negative values (log1p of values < -1 gives NaN)
#   - not binary 0/1 flags
safe_skewed_cols = [
    col for col in skewed_cols
    if col != 'review_scores_rating'
    and df[col].min() >= 0
    and df[col].nunique() > 2
]
st.write(f"Features to transform (high skewness): {safe_skewed_cols}")

for col in safe_skewed_cols:
    df[col] = np.log1p(df[col])

st.success("Applied log1p transformation to skewed features.")

# ------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------
st.subheader("Feature Selection")
st.markdown("Selecting relevant features using correlation and domain knowledge.")

numerical_cols = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit', 'cleaning_fee',
    'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'number_of_stays',
    'host_response_rate', 'guests_included', 'price_per_person', 'price_per_bedroom',
    'total_fees', 'beds_per_bedroom', 'years_as_host', 'amenities_count',
    'superhost_experience', 'min_max_nights_ratio'
]
categorical_cols = [
    'host_is_superhost', 'host_identity_verified', 'instant_bookable', 'is_business_travel_ready',
    'cancellation_policy', 'room_type', 'property_type'
]

numerical_cols = [col for col in numerical_cols if col in df.columns]
categorical_cols = [col for col in categorical_cols if col in df.columns]

if not numerical_cols:
    st.error("No numerical feature columns were found in the dataset.")
    st.stop()

st.subheader("Correlation Analysis")
corr_matrix = df[numerical_cols + ['review_scores_rating']].corr()
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={"size": 8})
plt.title("Correlation Matrix", fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
st.pyplot(fig)

correlations = corr_matrix['review_scores_rating'].drop('review_scores_rating').dropna()
strong_corr_features = correlations[correlations.abs() > 0.1].index.tolist()

if not strong_corr_features:
    st.warning("No numerical feature has |correlation| > 0.1 with the target; using all numerical features.")
    strong_corr_features = numerical_cols

selected_features = strong_corr_features + categorical_cols
df = df.dropna(subset=selected_features + ['review_scores_rating'])

if df.empty:
    st.error("No rows left after dropping missing values in the selected features.")
    st.stop()

X = df[selected_features].copy()
y = df['review_scores_rating']
st.write("Selected Features:", selected_features)

numerical_cols = [col for col in strong_corr_features if col in X.columns]
categorical_cols = [col for col in categorical_cols if col in X.columns]

# Make categorical columns uniform (bool/object mix -> str) for the encoder
for col in categorical_cols:
    X[col] = X[col].astype(str)

# ------------------------------------------------------------------
# Preprocessing pipeline
# ------------------------------------------------------------------
st.subheader("Preprocessing Pipeline")
st.markdown("Scaling numerical features and encoding categorical features.")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ------------------------------------------------------------------
# Model selection and hyperparameter tuning
# ------------------------------------------------------------------
st.subheader("Model Selection and Hyperparameter Tuning")
model_name = st.selectbox(
    "Select a regression model",
    ("Linear Regression", "Ridge Regression", "Random Forest",
     "LightGBM Regressor", "Support Vector Regression (SVR)")
)

if model_name == "Linear Regression":
    st.info("Linear Regression has no major hyperparameters to tune.")
    model = LinearRegression()

elif model_name == "Ridge Regression":
    alpha = st.slider("Ridge Regularization Strength (alpha)", 0.01, 10.0, 1.0)
    model = Ridge(alpha=alpha)

elif model_name == "Random Forest":
    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100)
    max_depth = st.slider("Maximum Tree Depth (max_depth)", 1, 50, 10)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

elif model_name == "LightGBM Regressor":
    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100)
    learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.1)
    model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

elif model_name == "Support Vector Regression (SVR)":
    C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    epsilon = st.slider("Epsilon (epsilon-insensitive tube)", 0.01, 1.0, 0.2)
    kernel = st.selectbox("Kernel Type", ("rbf", "linear", "poly", "sigmoid"))
    model = SVR(C=C, epsilon=epsilon, kernel=kernel)

model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)

# ------------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------------
st.subheader("Model Evaluation")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
st.subheader("Visualizations")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Review Scores")
ax.set_ylabel("Predicted Review Scores")
ax.set_title(f"{model_name}: Actual vs Predicted")
ax.grid(True)
st.pyplot(fig)

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, residuals, alpha=0.5, color='green')
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel("Predicted Review Scores")
ax.set_ylabel("Residuals")
ax.set_title(f"{model_name}: Residual Plot")
ax.grid(True)
st.pyplot(fig)

if model_name == "Random Forest":
    onehot_names = list(
        preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    )
    feature_names = numerical_cols + onehot_names
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    ax.set_title("Top 10 Feature Importances (Random Forest)")
    st.pyplot(fig)

# ------------------------------------------------------------------
# Conclusion
# ------------------------------------------------------------------
st.subheader("Conclusion")
st.markdown(f"""
The {model_name} model achieved an R² score of {r2:.2f}, indicating {r2 * 100:.0f}% of the variance in review scores is explained.

Key findings:
- Features such as price_per_person, host_is_superhost, and number_of_reviews may influence guest satisfaction (check the correlation heatmap and feature importances above).
- The mean residual is {residuals.mean():.3f}; values close to 0 suggest the errors are not systematically biased.

Next steps: experiment with additional features, tune hyperparameters, and test other models.
""")