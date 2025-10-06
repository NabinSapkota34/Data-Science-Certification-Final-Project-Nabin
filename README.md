# ODI Batsman Career Runs Prediction 🏏

## Objective
The goal of this project is to build a complete data science application from scratch. It involves creating a predictive model to estimate an ODI batsman's total career runs and deploying it as an interactive web application on `localhost` using Streamlit. This project covers the entire data science lifecycle as per the task requirements.

---

## 🚀 How to Run This Project

Follow these instructions to get the project running on your local machine.

### **Prerequisites**
* Python 3.7+
* `pip` (Python package installer)

### **Step 1: Set Up the Project Folder**
This is my folder structure
```
ds_final_project/
│
├── odb.csv
├── model_training.py
├── app.py
├── requirements.txt
└── README.md
```

### **Step 2: Create a Virtual Environment (Highly Recommended)**
Open your terminal or command prompt, navigate to the `ds_final_project` folder, and run the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **Step 3: Install Required Libraries**
Install all project dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### **Step 4: Train the Predictive Model**
Run the `model_training.py` script. This will perform all the data science tasks (cleaning, EDA, training) and create a file named `odi_runs_predictor.pkl`, which is your trained model.

```bash
python model_training.py
```
You will see the output of each step printed in your terminal, and two plot images (`runs_distribution.png` and `correlation_heatmap.png`) will be saved to your folder.

### **Step 5: Launch the Streamlit Web Application**
Now, run the `app.py` script to start the web application.

```bash
streamlit run app.py
```
This command will automatically open a new tab in your web browser at `http://localhost:8501`, where you can interact with your model.

---

## 📂 Project Deliverables Checklist

-   [x] **Python Script (`model_training.py`)**: Contains data loading, cleaning, EDA, feature selection, model training, and evaluation.
-   [x] **Streamlit Web Application (`app.py`)**: Provides an interactive UI for user inputs and displays model predictions.
-   [x] **GitHub Repository Files**: Includes project code, `README.md`, and `requirements.txt`.

---

## Git Integration for Version Control

To track your project with Git and upload it to GitHub:

1.  **Initialize Git**: In the project folder, run `git init`.
2.  **Create a `.gitignore` file**: Create a file named `.gitignore` and add the following lines to exclude the virtual environment and other temporary files from your repository.
    ```
    venv/
    __pycache__/
    *.pkl
    *.png
    ```
3.  **Add and Commit**: Stage and commit your initial project files.
    ```bash
    git add .
    git commit -m "Initial commit: Completed end-to-end data science project"
    ```
4.  **Create a GitHub Repository**: Go to [GitHub](https://github.com), create a new repository, and follow the instructions to link your local project and push your code.