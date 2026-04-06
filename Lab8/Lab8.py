import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and train on Breast Cancer dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

# Set random_state for reproducibility
bc_model = DecisionTreeClassifier(max_depth=4, random_state=42)
bc_model.fit(X_cancer, y_cancer)
print("Model trained successfully.")
# Step 2: Custom sample dataset (for job offer prediction)
sample_data = pd.DataFrame({
    'cgpa': [9.2, 8.5, 9.0, 7.5, 8.2, 9.1, 7.8, 9.3, 8.4, 8.6],
    'interactiveness': ['yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes'],
    'practical_knowledge': ['verygood', 'good', 'average', 'average', 'good', 'good', 'good', 'verygood', 'good', 'average'],
    'communication': ['good', 'moderate', 'poor', 'good', 'moderate', 'moderate', 'poor', 'good', 'good', 'good'],
    'job_offer': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']
})
# Encode categorical features
label_encoders = {}
for column in ['interactiveness', 'practical_knowledge', 'communication', 'job_offer']:
    le = LabelEncoder()
    sample_data[column] = le.fit_transform(sample_data[column])
    label_encoders[column] = le

X_sample = sample_data.drop('job_offer', axis=1)
y_sample = sample_data['job_offer']

# Step 3: Train a Decision Tree on the sample data
sample_model = DecisionTreeClassifier(max_depth=4, random_state=42)
sample_model.fit(X_sample, y_sample)

# Step 4: Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(sample_model, feature_names=X_sample.columns,
          class_names=label_encoders['job_offer'].classes_,
          filled=True, rounded=True)
plt.title("Decision Tree for Job Offer Prediction")
plt.show()

# Step 5: Test the model with a new sample input
test_sample = pd.DataFrame([{
    'cgpa': 6.5,
    'interactiveness': 'yes',
    'practical_knowledge': 'good',
    'communication': 'good'
}])

# Encode the test input using the same label encoders
for column in ['interactiveness', 'practical_knowledge', 'communication']:
    test_sample[column] = label_encoders[column].transform(test_sample[column])

prediction = sample_model.predict(test_sample)
predicted_label = label_encoders['job_offer'].inverse_transform(prediction)
print("Predicted Job Offer for test sample:", predicted_label[0])
