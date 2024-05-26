# Medicine Recommendation System 🏥

## Project Overview

This project aims to create a medicine recommendation system based on symptoms provided by the user. The system is built using machine learning models trained on a dataset of symptoms and their corresponding diagnoses. The frontend is designed using Bootstrap for an intuitive user interface.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Frontend](#frontend)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset contains various symptoms and the corresponding prognosis (diagnosis). It includes 4920 instances and 133 columns, where each column represents a symptom (binary value) and the prognosis.

## Preprocessing

1. **Loading the Dataset:**
   ```python
   df = pd.read_csv("E:\\Medicine Recommendation System\\datasets\\Training.csv")
   ```

2. **Splitting the Dataset:**
   ```python
   X = df.drop('prognosis', axis=1)
   Y = df['prognosis']

   le = LabelEncoder()
   le.fit(Y)
   Y = le.transform(Y)

   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
   ```

## Model Training

Several machine learning models were trained and evaluated to find the best-performing model:

- Support Vector Classifier (SVC)
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- MultinomialNB

Example code for training and evaluating the models:
```python
models = {
    "SVC": SVC(kernel='linear'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MultinomialNB": MultinomialNB()
}

for model_name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    cm = confusion_matrix(Y_test, predictions)
    print(f"{model_name} Accuracy:  {accuracy}")
    print(f"{model_name} Confusion Matrix:")
    print(np.array2string(cm, separator=', '))
```

## Evaluation

All models achieved an accuracy of 100% on the test set. The confusion matrices also indicate perfect predictions for all classes.

## Frontend

The frontend is built using Bootstrap to provide a user-friendly interface. Users can input their symptoms, and the system will predict the possible diagnosis.

## Usage

1. **Training the Model:**
   ```python
   svc = SVC(kernel='linear')
   svc.fit(X_train, Y_train)
   pickle.dump(svc, open('svc.pkl', 'wb'))
   ```

2. **Loading the Model:**
   ```python
   svc = pickle.load(open('svc.pkl', 'rb'))
   ypredict = svc.predict(X_test)
   print(f"Predicted Label :  {ypredict}")
   print(f"Actual Label:  {Y_test}")
   ```

3. **Running the Frontend:**
   - Ensure you have a web server set up to serve the HTML files.
   - Open the main HTML file in a browser to interact with the system.

## Dependencies

- pandas
- numpy
- scikit-learn
- pickle
- Bootstrap (for the frontend)


## Contributing

We welcome contributions! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
