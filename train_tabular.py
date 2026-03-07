import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_tabular_model():
    print("Loading Wisconsin Breast Cancer Dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
             'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
             'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    df = pd.read_csv(url, names=names)

    # Preprocessing
    print("Preprocessing data...")
    # Handle missing values (marked as '?')
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df['bare_nuclei'] = df['bare_nuclei'].astype(int)

    # Drop ID column
    df.drop('id', axis=1, inplace=True)

    # Features and Target
    X = df.drop('class', axis=1)
    y = df['class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling and Model Training in a Pipeline-like fashion
    # (The app loads model.pkl directly, which in the original repo was often a combined object or just the model)
    # However, app.py line 108 says: model = pickle.load(open('model.pkl', 'rb'))
    # And original notebook used a pipeline often. 
    # Let's ensure we save an object that app.py can use.
    # Looking at app.py lines 170-188, it expects the model to have a .predict() method.
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training SVM model...")
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X_train_scaled, y_train)

    # Evaluation
    predictions = svm_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, predictions))

    # Save model and scaler
    # Since app.py only loads 'model.pkl', we should ideally save a Pipeline or 
    # handle scaling inside the predict logic if app.py doesn't do it.
    # Looking at app.py, it DOES NOT scale the input before calling model.predict().
    # This means 'model.pkl' MUST be a Pipeline or we must update app.py.
    # Original app.py relied on a model.pkl that likely didn't expect scaling or was trained on raw data?
    # Actually, the notebook shows StandardScalar.
    
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline([
        ('scaler', scaler),
        ('svm', svm_model)
    ])

    print("Saving model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(full_pipeline, f)
    
    print("Retraining of tabular model complete.")

if __name__ == "__main__":
    train_tabular_model()
