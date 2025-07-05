import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Preprocess: Handle missing values (if any), convert categorical variables
    label_encoder = LabelEncoder()
    data['Type'] = label_encoder.fit_transform(data['Type'])

    return data, label_encoder

def feature_engineering(data):
    # Here, you can add any new features if needed
    return data

def train_decision_tree(X_train, y_train):
    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=0, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    # Predict and evaluate
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Save metrics to a file
    with open('deliverables/metrics.txt', 'w') as f:
        f.write(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}")

    return precision, recall, f1

def main():
    # File path to the dataset
    file_path = 'data/fraud_detection.csv'

    # Load and preprocess data
    data, label_encoder = load_and_preprocess_data(file_path)

    # Feature engineering
    data = feature_engineering(data)

    # Define features and target
    X = data[['Amount', 'Type']]
    y = data['Is Fraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    clf = train_decision_tree(X_train, y_train)

    # Save model
    joblib.dump(clf, 'deliverables/decision_tree_model.pkl')

    # Also save a text representation of the model
    model_summary = str(clf.get_params())
    with open('deliverables/model_summary.txt', 'w') as f:
        f.write(model_summary)

    # Evaluate model
    precision, recall, f1 = evaluate_model(clf, X_test, y_test)

    # Print evaluation metrics
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # Deliverable 3: Recommendations
    recommendations = """
    Recommendations to improve fraud detection accuracy:
    1. Use more sophisticated models like Random Forest or XGBoost.
    2. Perform more extensive feature engineering.
    3. Collect more data to improve model robustness.
    """
    with open('deliverables/recommendations.txt', 'w') as f:
        f.write(recommendations)

    print("Deliverables have been generated.")

if __name__ == "__main__":
    main()
