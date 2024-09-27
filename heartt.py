import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filename):
    """Loads the heart disease dataset from a CSV file."""
    df = pd.read_csv(filename)
    return df

def explore_data(df):
    """Performs exploratory data analysis (EDA)."""
    print(df.head())
    print(df.info())
    print(df.describe())

    # Correlations
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    # Distribution of numerical features
    df.hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()

    # Distribution of target variable by other features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for feature in categorical_features:
        sns.countplot(x=feature, hue='target', data=df)
        plt.show()

def preprocess_data(df):
    """Preprocesses the data for modeling."""
    # Handle missing values (if any)
    # df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values

    # Feature engineering (consider creating new features)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    """Builds a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model's performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)

def create_dashboard(df, model):  # Placeholder for future dashboard development
    """Creates a dashboard to visualize key insights (to be implemented)."""
    # ... Implement dashboard creation using libraries like Plotly, Dash, etc.
    # ... Consider visualizing correlations, feature importance, risk factors, etc.
    pass

if __name__ == "__main__":
    filename = "Heart Disease data.csv"  # Replace with your filename
    df = load_data(filename)
    explore_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df.copy())
    model = build_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    create_dashboard(df.copy(), model)  # Pass a copy to avoid modifying original data