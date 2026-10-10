# Iris Flower Classification API

A machine learning classification project that predicts the species of an Iris flower using **Logistic Regression**, **FastAPI**, and a saved model (`.pkl`). The API can be tested locally and deployed to Render, with source code maintained on GitHub.

## Project Overview

This project uses the Iris dataset, which contains 150 records of Iris flowers. Based on four flower measurements, the trained machine learning model predicts one of three species.

The main objective is to understand the complete machine learning deployment workflow, from model training to building a REST API and deploying it to the cloud.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* StandardScaler
* Joblib
* FastAPI
* Uvicorn
* Git and GitHub
* Render

## Dataset Information

The dataset contains 150 records and five columns.

| Feature        | Description                                   |
| -------------- | --------------------------------------------- |
| `sepal_length` | Length of the sepal in centimeters            |
| `sepal_width`  | Width of the sepal in centimeters             |
| `petal_length` | Length of the petal in centimeters            |
| `petal_width`  | Width of the petal in centimeters             |
| `species`      | Target variable representing the Iris species |

### Target Classes

The model predicts one of the following species:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

## Machine Learning Workflow

1. Load the Iris dataset using Pandas.
2. Separate the input features and target variable.
3. Split the dataset into training and testing sets.
4. Scale the input features using `StandardScaler`.
5. Train a Logistic Regression classification model.
6. Evaluate the model using accuracy and a classification report.
7. Save the complete preprocessing and model pipeline as a `.pkl` file.
8. Load the saved pipeline in FastAPI.
9. Accept input measurements through a REST API.
10. Return the predicted species and confidence score.
11. Push the project to GitHub and deploy it on Render.

## Project Structure

```text
iris-classification-api/
│
├── data/
│   └── iris.csv
│
├── model/
│   └── iris_model.pkl
│
├── train_model.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/iris-classification-api.git
cd iris-classification-api
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Ensure that `data/iris.csv` exists, then run:

```bash
python train_model.py
```

This trains the Logistic Regression pipeline, evaluates its performance, and saves the trained pipeline to:

```text
model/iris_model.pkl
```

### 5. Start the FastAPI application

After generating the model file, run:

```bash
uvicorn main:app --reload
```

The application will start locally.

## API Endpoints

| Method | Endpoint   | Description                                     |
| ------ | ---------- | ----------------------------------------------- |
| GET    | `/`        | Displays a welcome message                      |
| GET    | `/health`  | Checks whether the API is running               |
| POST   | `/predict` | Predicts the Iris flower species                |
| GET    | `/docs`    | Opens the interactive Swagger API documentation |

### Local API URL

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Making a Prediction

Send a POST request to `/predict` with the following JSON input:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Example Response

```json
{
  "predicted_species": "Iris-setosa",
  "confidence": 0.97
}
```

The confidence value above is illustrative; the actual value depends on the trained model.

You can test this endpoint using the interactive Swagger documentation at `/docs`.

## Model Evaluation

The model is evaluated using:

* **Accuracy:** Proportion of correct predictions.
* **Precision:** Measures the correctness of positive predictions for each class.
* **Recall:** Measures how many actual instances of each class are correctly identified.
* **F1-score:** Harmonic mean of precision and recall.

The evaluation results are printed when `train_model.py` runs. Actual performance metrics should be recorded from the executed training script rather than assumed in advance.

## Why Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for classification. Although its name contains the word regression, it predicts class probabilities and assigns observations to classes.

In this project:

* The dataset contains three target classes.
* `StandardScaler` scales the input features.
* Logistic Regression learns the relationship between measurements and species.
* Scikit-learn handles the multiclass classification automatically.

The scaler and classifier are saved together in a Scikit-learn pipeline, ensuring that new inputs receive the same preprocessing during prediction.

## Deployment on Render

### 1. Push the project to GitHub

```bash
git add .
git commit -m "Add Iris classification API"
git push origin main
```

### 2. Create a Render web service

1. Sign in to [Render](https://render.com/).
2. Create a new Web Service.
3. Connect your GitHub repository.
4. Configure the build and start commands.

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

5. Deploy the service and review the deployment logs.
6. Open the public service URL once deployment succeeds.

After deployment, test `/`, `/health`, `/predict`, and `/docs` using your Render URL.

## Important Notes

* The model file must exist before starting the API.
* The `data/iris.csv` file must be present when training the model.
* Keep the training and inference feature names and ordering consistent.
* Keep the `scikit-learn` version compatible between model training and deployment.
* Only load trusted `.pkl` model files because deserializing untrusted files can execute arbitrary code.
* The API's confidence score represents the highest predicted class probability; it is not a guarantee of correctness.

## Learning Outcomes

By completing this project, you will learn how to:

* Build a multiclass classification model.
* Create a preprocessing and machine learning pipeline.
* Save and load a trained model using Joblib.
* Develop prediction endpoints using FastAPI.
* Test REST APIs using Swagger UI.
* Manage project dependencies using `requirements.txt`.
* Track source code using Git and GitHub.
* Deploy a machine learning API using Render.

## Future Improvements

* Add input validation and better error handling.
* Add automated tests for the prediction endpoint.
* Record evaluation metrics and a confusion matrix.
* Add continuous integration and automated deployment.
* Deploy a more complex classification project, such as customer churn prediction.

---

**Author:** Your Name

**Project:** Iris Flower Classification API

**Purpose:** Learning end-to-end machine learning deployment with Logistic Regression, FastAPI, GitHub, and Render.
