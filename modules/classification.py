from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_classifier(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def classify_sample(band_averages):
    # Load your training data and labels from dataset.csv
    dataset_path = 'C:/Users/User/Desktop/WQA/WQA/dataset/dataset.csv'
    data = pd.read_csv(dataset_path)

    # Drop the 'Sample Name' column if it exists
    data.drop(columns=['Sample Name'], inplace=True, errors='ignore')

    X_train = data.drop('Class', axis=1)  # Assuming the label column is named 'label'
    y_train = data['Class']

    # Train the classifier
    classifier = train_classifier(X_train, y_train)

    # Make predictions using the trained classifier
    prediction = classifier.predict([band_averages])

    return prediction[0]