import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

log_reg = joblib.load(r'../models/89_logistic_regression_model.pkl')

dataset = pd.read_csv('../data/preprocessed_data.csv',encoding='ISO-8859-1')
print(f'Dataset loaded successfully!')
dataset = dataset.drop(['Unnamed: 0'], axis=1).dropna(subset=['Text'])

# Load the pre-fitted vectorizer
tfidf = TfidfVectorizer()
print(f'TFIDF Vectorizer created!')
print(f'Transforming...')
X = tfidf.fit_transform(dataset['Text'])
print(f'X column vectorized!')

y = dataset['Human']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Test data created!')

# predict using the loaded model
predictions = log_reg.predict(X_test)

# evaluate model accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy}")

# evaluation metrics
conf_matrix = confusion_matrix(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
fone = f1_score(y_test, predictions)
print()
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {fone}")


# plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

