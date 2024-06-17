A ML Approach Using Statistical Models for Early Detection of Cardiac Arrest in Newborn Babies in the Cardiac Intensive Care Unit


This is a Tkinter-based Python application that allows users to load a CSV file, preprocess the data, and run different machine learning algorithms to predict the likelihood of cardiac arrest in newborn babies. The application supports Logistic Regression, Decision Tree, Random Forest,Naive bayes,KNN and SVM algorithms. The results, including accuracy, recall, and specificity, are displayed within the application and visualized using bar plots.

Features>
Load and preprocess CSV data.
Encode categorical variables.
Split data into training and testing sets.
Run and evaluate multiple machine learning algorithms:
Logistic Regression
Decision Tree
Random Forest
Naive Bayes
SVM
Display and visualize results.

Requirements>
Python 3.x
Tkinter
pandas
scikit-learn
matplotlib

Run the application:
(in bash)
python app.py


Follow these steps within the application:
Click "Browse" to select a CSV file containing the dataset.
Enter the name of the target column when prompted.
Click "Preprocess Data" to prepare the dataset for training.
Select one of the algorithms (Logistic Regression, Decision Tree, Random Forest,KNN,Naive Bayes or SVM) to train and evaluate the model.
View the results and performance metrics in the text box and the plot.

Code Overview>
app.py>
Cardiac_arrest Class: This class defines the main application window and its functionalities.
__init__(self, root): Initializes the main application window, including buttons, text fields, and plotting canvas.
load_file(self): Loads the selected CSV file and verifies the target column.
preprocess_data(self): Preprocesses the data by handling missing values, encoding categorical variables, and splitting into training and testing sets.
evaluate_model(self, model): Evaluates the trained model using accuracy, recall, and specificity metrics.
run_ml(self, model_name): Runs the selected machine learning algorithm and displays the results.
display_results(self, model_name, accuracy, recall, specificity): Displays the evaluation results in the text widget.
plot_results(self, model_name, accuracy, recall, specificity): Plots the evaluation metrics using a bar chart.


##OR##


# Execution Instructions

# Python version 3.11.9

To create a virtual environment and install requirements in Python 3.11.9 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:


cd C:\path\to\project

Create a new virtual environment using the venv module:


python -m venv myenv

Activate the virtual environment:
myenv\Scripts\activate


Install the project requirements using pip:
pip install -r requirements.txt

### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

cd /path/to/project

Create a new virtual environment using the venv module:

python3.10 -m venv myenv


Activate the virtual environment:
source myenv/bin/activate

Install the project requirements using pip:
pip install -r requirements.txt

These instructions assume you have Python 3.10.4 installed and added to your system's PATH variable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.11.9. Specify the version using the -p or --python flag. Follow the instructions below:

For Windows:
Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:

cd C:\path\to\project

Create a new virtual environment using the Python Launcher:

py -3.11 -m venv myenv

Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:


myenv\Scripts\activate


Install the project requirements using pip:

pip install -r requirements.txt


### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

cd /path/to/project

Create a new virtual environment using the Python Launcher:


python3.11 -m venv myenv


Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:

source myenv/bin/activate


Install the project requirements using pip:

pip install -r requirements.txt


By specifying the version using py -3.11 or python3.11, you can ensure that the virtual environment is created using Python 3.11.9 specifically, even if you have other Python versions installed.





