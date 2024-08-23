# My-Disease-Predictor
> Predicting diseases using machine learning

This project is a web-based application for predicting the likelihood of stroke and diabetes using user-provided medical information.


## Overview:
My-Disease-Predictor is a simple web application built using Flask and machine learning algorithms to predict the likelihood of diseases like stroke and diabetes. Users can input various health metrics such as age, gender, blood pressure, cholesterol levels, BMI, and more. The application then processes these inputs and provides a prediction of the user's risk for these conditions.


![](Homepage.jpg)
![Home page](https://github.com/user-attachments/assets/2312ddbe-ca3a-456a-b2e9-dcf8c62b41cb)


## Repository Structure:
`app.py`:  The main Flask application file that serves as the engine for this project. It handles routing, data input, prediction, and rendering of HTML templates.

`static`/: Contains the static files like CSS, images, etc.

`style.css`: Adds styling and enhances the look of the application.

`templates`/: Contains HTML files that define the structure and behavior of the web app.

`ipynb`: Contains the pre-trained machine learning models.

`requirements.txt`: Lists all the Python dependencies required for the project.




#### Warning: Please note that this Disease prediction application is intended to be used solely as an assessment tool and not as a substitute for professional medical advice and diagnosis. The predictions generated by this app are based on statistical models and available data, which means they may not always be accurate. It is essential to consult with a qualified healthcare provider to interpret the results and to obtain personalized medical advice


## Getting started:
To run the My-Disease-Predictor application on your local machine, follow these steps:

#### Clone the Repository
First, clone the repository to your local machine using Git:
```
git clone https://github.com/SBABAJ200/My-Disease-Predictor.git

```
#### Set up a virtual enviroment

```
python -m venv venv

```
```
venv/Scripts/activate
```


## Installation:
* Execute the command: `python app.py`
* Open http://127.0.0.1:5000/ in your  web browser


## Future Work:
* Improve the model accuracy with more advanced algorithms
* Add more features to the user interface for better user experience
