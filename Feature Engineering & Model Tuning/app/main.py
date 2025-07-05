from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def load_dataset():
    # Construct the absolute path to the CSV file
    script_dir = os.path.dirname(__file__)  # This gets the directory where the script is located
    file_path = os.path.join(script_dir, '..', 'data', 'student_scores.csv')
    absolute_path = os.path.abspath(file_path)

    # Check if the file exists
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"The file {absolute_path} does not exist.")

    # Load the dataset
    df = pd.read_csv(absolute_path)
    return df

def perform_feature_engineering(df):
    if 'Total_Score' not in df.columns:
        df['Total_Score'] = df[['Math', 'Science', 'English', 'History']].sum(axis=1)
    return df

def hyperparameter_tuning(df):
    X = df[['Math', 'Science', 'English', 'History']]
    y = df['Total_Score'].apply(lambda x: 1 if x >= 200 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return best_params, best_score, accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        df = load_dataset()
        df = perform_feature_engineering(df)
        best_params, best_score, accuracy = hyperparameter_tuning(df)

        return jsonify({
            'best_params': best_params,
            'best_score': best_score,
            'accuracy': accuracy,
            'df_head': df.head().to_html()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)