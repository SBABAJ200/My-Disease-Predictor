from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer

# Create application
app = Flask(__name__)

# Load and prepare the stroke (heart disease) model
stroke_dataset_path = 'healthcare-dataset-stroke-data.csv'  
stroke_dataframe = pd.read_csv(stroke_dataset_path)

# Prepare stroke features and labels
x_stroke = stroke_dataframe.drop(['stroke'], axis=1)
Y_stroke = stroke_dataframe['stroke']

# Manually map categorical variables
stroke_dataframe['gender'] = stroke_dataframe['gender'].map({'Male': 1, 'Female': 0})
stroke_dataframe['ever_married'] = stroke_dataframe['ever_married'].map({'Yes': 1, 'No': 0})
stroke_dataframe['work_type'] = stroke_dataframe['work_type'].map({
    'Private': 0,
    'Self-employed': 1,
    'Govt_job': 2,
    'children': 3,
    'Never_worked': 4
})
stroke_dataframe['Residence_type'] = stroke_dataframe['Residence_type'].map({'Urban': 1, 'Rural': 0})
stroke_dataframe['smoking_status'] = stroke_dataframe['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3
})


# Update features after mapping
x_stroke = stroke_dataframe.drop(['stroke'], axis=1)

# Save stroke column names after mapping for later use
stroke_column_names = x_stroke.columns

# Split the stroke data into training and test sets
x_stroke_train, x_stroke_test, Y_stroke_train, Y_stroke_test = train_test_split(x_stroke, Y_stroke, test_size=0.2, random_state=42)

# Create an imputer object with a strategy to fill missing values
imputer = SimpleImputer(strategy='mean')  # You can change 'mean' to 'median', 'most_frequent', etc.

# Fit the imputer on the training data and transform both training and test data
x_stroke_train = imputer.fit_transform(x_stroke_train)
x_stroke_test = imputer.transform(x_stroke_test)

# Create and train the BernoulliNB classifier for stroke prediction
stroke_scaler = StandardScaler()
x_stroke_train = stroke_scaler.fit_transform(x_stroke_train)
x_stroke_test = stroke_scaler.transform(x_stroke_test)
stroke_model = BernoulliNB()
stroke_model.fit(x_stroke_train, Y_stroke_train)

# Evaluate the stroke model
Y_stroke_pred = stroke_model.predict(x_stroke_test)
stroke_accuracy = accuracy_score(Y_stroke_test, Y_stroke_pred)
print(f"Stroke Model accuracy: {stroke_accuracy * 100:.2f}%")

# Save the trained stroke model using pickle
stroke_model_path = 'Stroke_prediction_model.pkl'
with open(stroke_model_path, 'wb') as file:
    pickle.dump(stroke_model, file)

# Load and prepare the diabetes model
diabetes_dataset_path = r'C:\Users\sbaba\OneDrive\Documents\Disease Prediction\Dataset of Diabetes .csv' 
diabetes_dataframe = pd.read_csv(diabetes_dataset_path)

# Encode Gender for diabetes dataset
diabetes_label_encoder = LabelEncoder()
diabetes_dataframe['Gender'] = diabetes_label_encoder.fit_transform(diabetes_dataframe['Gender'])

# Prepare diabetes features and labels
x_diabetes = diabetes_dataframe.drop(['CLASS'], axis=1)
Y_diabetes = diabetes_dataframe['CLASS']

# One-hot encode diabetes dataset if necessary
x_diabetes = pd.get_dummies(x_diabetes)

# Save the diabetes column names
diabetes_column_names = x_diabetes.columns

# Split the diabetes dataset
x_diabetes_train, x_diabetes_test, Y_diabetes_train, Y_diabetes_test = train_test_split(x_diabetes, Y_diabetes, test_size=0.2, random_state=42)

# Create and train the BernoulliNB classifier for diabetes
diabetes_scaler = StandardScaler()
x_diabetes_train = diabetes_scaler.fit_transform(x_diabetes_train)
x_diabetes_test = diabetes_scaler.transform(x_diabetes_test)
diabetes_model = BernoulliNB()
diabetes_model.fit(x_diabetes_train, Y_diabetes_train)

# Evaluate the diabetes model
Y_diabetes_pred = diabetes_model.predict(x_diabetes_test)
diabetes_accuracy = accuracy_score(Y_diabetes_test, Y_diabetes_pred)
print(f"Diabetes Model accuracy: {diabetes_accuracy * 100:.2f}%")

# Save the trained diabetes model using pickle
diabetes_model_path = 'Diabetes_prediction_model.pkl'
with open(diabetes_model_path, 'wb') as file:
    pickle.dump(diabetes_model, file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/heart_disease_prediction')
def heart_disease_prediction():
    return render_template('heart_disease_prediction.html')

@app.route('/diabetes_prediction')
def diabetes_prediction():
    return render_template('diabetes_prediction.html')

@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
    try:
        print("Form Data:", request.form)

        # Extract form data from request.form
        form_data = request.form.to_dict()
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Prepare the input data
        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
        input_features_df = pd.DataFrame(input_data, columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

        # Reindex the DataFrame to ensure it has the same columns as the training data
        input_features_df = input_features_df.reindex(columns=stroke_column_names, fill_value=0)

        # Convert DataFrame to numpy array for prediction
        input_features_np = stroke_scaler.transform(input_features_df.to_numpy())

        # Predict using the model
        prediction_proba = stroke_model.predict_proba(input_features_np)

        # Manually determine the prediction based on the highest probability
        max_index = np.argmax(prediction_proba, axis=1)[0]
        class_mapping = {0: 'No Stroke', 1: 'Stroke'}
        output = class_mapping[max_index]

         # Include health website links if the prediction is "Diabetic"
        links = []
        if output == 'Stroke':
            links = [
                "https://www.health.harvard.edu/womens-health/8-things-you-can-do-to-prevent-a-stroke",

                "https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/heart-disease-and-stroke",
                
                "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke"
            ]


        # Create a response with prediction and probabilities
        response = {
            "prediction": output,
            "probabilities": {
                "No Stroke": f"{prediction_proba[0][0] * 100:.2f}%",
                "Stroke": f"{prediction_proba[0][1] * 100:.2f}%"
            },
            "links":links
        }

        print("Prediction probabilities:", prediction_proba)

        return render_template('Heart_disease_prediction.html', prediction_text=f"Prediction: {response['prediction']}", probabilities=response['probabilities'],links=response['links'],form_data=form_data)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/diabetes_prediction')
def Diabetes_prediction():
    return render_template('diabetes_prediction.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        print("Form Data:", request.form)  # Add this line to see the incoming form data

        # Extract form data from request.form and apply clipping
        form_data = request.form.to_dict()
        gender = 0 if request.form['Gender'] == 'Male' else 1  # 1 = Male, 0 = Female
        age = np.clip(float(request.form['Age']), 0, 120)  # Age between 0 and 120 years
        urea = np.clip(float(request.form['Urea']), 10, 100)  # Urea between 10 and 100 mg/dL
        hba1c = np.clip(float(request.form['HbA1c']), 3.0, 15.0)  # HbA1c between 3.0% and 15.0%
        creatinine_ratio = np.clip(float(request.form['Creatinine_ratio']), 0, 1000)  # Creatinine ratio between 0.5 and 10 mg/dL
        cholesterol = np.clip(float(request.form['Cholesterol']), 100, 400)  # Cholesterol between 100 and 400 mg/dL
        ldl = np.clip(float(request.form['LDL']), 0, 100)  # LDL between 0 and 100 mg/dL
        bmi = np.clip(float(request.form['BMI']), 10, 50)  # BMI between 10 and 50
        vldl = np.clip(float(request.form['VLDL']), 2, 50)  # VLDL between 2 and 50 mg/dL
        triglycerides = np.clip(float(request.form['Triglycerides']), 0.5, 500)  # Triglycerides between 50 and 500 mg/dL
        hdl = np.clip(float(request.form['HDL']), 0, 100)  # HDL between 20 and 100 mg/dL

        # Prepare the input data
        input_data = np.array([[gender, age, urea, hba1c, creatinine_ratio, cholesterol, ldl, bmi, vldl, triglycerides, hdl]])
        input_features_df = pd.DataFrame(input_data, columns=['Gender', 'Age', 'Urea', 'HbA1c', 'Creatinine_ratio', 'Cholesterol', 'LDL', 'BMI', 'VLDL', 'Triglycerides', 'HDL'])

        # Reindex the DataFrame to ensure it has the same columns as the training data
        input_features_df = input_features_df.reindex(columns=diabetes_column_names, fill_value=0)

        # Convert DataFrame to numpy array for prediction
        input_features_np = diabetes_scaler.transform(input_features_df.to_numpy())

        # Predict using the model
        prediction_proba = diabetes_model.predict_proba(input_features_np)
        print("Prediction probabilities:", prediction_proba)

        # Aggregate probabilities for class 1 (Diabetic) and class 2 (Predict Diabetic)
        diabetic_proba = prediction_proba[0][1] + prediction_proba[0][2] if prediction_proba.shape[1] > 2 else prediction_proba[0][1]

        # Manually determine the prediction based on the highest probability
        max_index = np.argmax([prediction_proba[0][0], diabetic_proba])  # only 2 classes now
        print("Max Index:", max_index)  # Log the predicted index

        class_mapping = {0: 'Non-Diabetic', 1: 'Diabetic'}
        output = class_mapping[max_index]

         # Include health website links if the prediction is "Diabetic"
        links = []
        if output == 'Diabetic':
            links = [
                "https://www.diabetes.org/",
                "https://www.cdc.gov/diabetes/index.html",
                "https://www.who.int/news-room/fact-sheets/detail/diabetes"
            ]

        # Create a response with prediction and probabilities
        response = {
            "prediction": output,
            "probabilities": {
                "Non Diabetic": f"{prediction_proba[0][0] * 100:.2f}%",
                "Diabetic": f"{diabetic_proba * 100:.2f}%"
            },
            "links": links
        }

        print("Prediction probabilities:", prediction_proba)

        return render_template('diabetes_prediction.html', prediction_text=f"Prediction: {response['prediction']}", probabilities=response['probabilities'],links=response['links'],form_data=form_data)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

