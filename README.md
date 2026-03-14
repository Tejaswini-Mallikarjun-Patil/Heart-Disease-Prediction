# ❤️ Heart Disease Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction of heart disease can help doctors take preventive actions and improve patient outcomes.

This project builds a **machine learning-based prediction system using an Artificial Neural Network (ANN)** to estimate the likelihood of heart disease based on patient medical parameters.

A **Streamlit web application** is developed to allow users to input patient data and receive real-time predictions.

---

# 🎯 Objective

The goal of this project is to:

* Predict the risk of heart disease using medical data
* Apply **Artificial Neural Networks (ANN)** for classification
* Deploy the trained model as an **interactive web application**
* Demonstrate practical use of **machine learning in healthcare**

---

# 🧠 Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| TensorFlow   | Deep learning framework        |
| Keras        | Neural network API             |
| NumPy        | Numerical computations         |
| Pickle       | Model preprocessing storage    |
| Streamlit    | Web application interface      |
| Scikit-learn | Data preprocessing and scaling |

---

# 📊 Dataset Description

The dataset contains **13 medical attributes** used to predict heart disease.

### Features Used

| Feature                 | Description                          |
| ----------------------- | ------------------------------------ |
| Age                     | Age of the patient                   |
| Sex                     | Gender of the patient                |
| Chest Pain Type         | Type of chest pain                   |
| Resting Blood Pressure  | Blood pressure at rest               |
| Cholesterol             | Serum cholesterol level              |
| Fasting Blood Sugar     | Blood sugar > 120 mg/dl              |
| Rest ECG                | Resting electrocardiographic results |
| Max Heart Rate          | Maximum heart rate achieved          |
| Exercise Induced Angina | Chest pain during exercise           |
| Oldpeak                 | ST depression induced by exercise    |
| Slope                   | Slope of peak exercise ST segment    |
| CA                      | Number of major vessels              |
| Thal                    | Thalassemia blood disorder indicator |

---

# 🏗 Model Architecture

The model is built using a **Sequential Artificial Neural Network**.

### Architecture

Input Layer:
13 Features

Hidden Layer 1:
8 neurons (ReLU activation)

Hidden Layer 2:
14 neurons (ReLU activation)

Output Layer:
1 neuron (Sigmoid activation)

### Model Diagram

```
Input Layer (13 features)
        ↓
Hidden Layer (8 neurons, ReLU)
        ↓
Hidden Layer (14 neurons, ReLU)
        ↓
Output Layer (1 neuron, Sigmoid)
```

---

# ⚙️ Model Implementation

### Model Creation

```python
classifier = Sequential()

classifier.add(Dense(activation="relu",
                     input_dim=13,
                     units=8,
                     kernel_initializer="uniform"))

classifier.add(Dense(activation="relu",
                     units=14,
                     kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid",
                     units=1,
                     kernel_initializer="uniform"))
```

### Model Compilation

```python
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

# 🔬 Activation Functions

### ReLU (Rectified Linear Unit)

Used in hidden layers to introduce non-linearity.

Formula:

```
ReLU(x) = max(0, x)
```

Advantages:

* Faster training
* Avoids vanishing gradient
* Improves learning efficiency

---

### Sigmoid

Used in the output layer for binary classification.

Formula:

```
Sigmoid(x) = 1 / (1 + e^-x)
```

Output range:

```
0 → 1
```

Example:

* 0.85 → High probability of heart disease
* 0.15 → Low probability

---

# 📉 Loss Function

The model uses **Binary Crossentropy**.

This loss function measures how far the predicted value is from the actual label in binary classification problems.

---

# ⚡ Optimizer

The **Adam optimizer** is used to update network weights efficiently.

Advantages:

* Faster convergence
* Adaptive learning rate
* Works well for deep learning models

---

# 🖥 Streamlit Web Application

A Streamlit interface allows users to:

* Enter patient medical details
* Process data through the trained ANN model
* Display prediction results instantly

### Prediction Output

The system returns:

* **High Risk of Heart Disease**
* **Low Risk of Heart Disease**

based on the probability predicted by the ANN model.

---

# 🚀 How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/heart-disease-ann.git
```

### 2️⃣ Navigate to the project folder

```
cd heart-disease-ann
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit application

```
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

```
heart-disease-ann
│
├── app.py
├── model.h5
├── scaler.pkl
├── dataset.csv
├── requirements.txt
└── README.md
```

---

# 📈 Future Improvements

Possible enhancements for this project include:

* Comparing multiple classifiers
* Hyperparameter tuning
* Model performance visualization
* Feature importance analysis
* Deployment on cloud platforms

---

# 🌍 Applications

This project can assist in:

* Early heart disease screening
* Medical decision support systems
* Healthcare data analysis
* AI-assisted diagnosis

---

# 👩‍💻 Author

**Tejaswini Patil**

Artificial Intelligence and Machine Learning Enthusiast

---

# ⭐ If you like this project

Consider giving the repository a **star ⭐ on GitHub**.
