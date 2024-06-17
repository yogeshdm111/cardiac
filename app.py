import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import math

class Cardiac_arrest:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Cardiac arrest in newborn babies")
        self.root.geometry("800x800")
        
        # Label and button to load the CSV file
        self.file_label = tk.Label(root, text="Select a CSV file:")
        self.file_label.pack()
        self.file_button = tk.Button(root, text="Browse", command=self.load_file)
        self.file_button.pack()
        
        # Button to preprocess data (disabled until a file is loaded)
        self.preprocess_button = tk.Button(root, text="Preprocess Data", command=self.preprocess_data, state=tk.DISABLED)
        self.preprocess_button.pack()
        
        # Button to perform EDA (disabled until a file is loaded)
        self.eda_button = tk.Button(root, text="Perform EDA", command=self.perform_eda, state=tk.DISABLED)
        self.eda_button.pack()
        
        # Frame to hold algorithm buttons
        self.algorithms_frame = tk.Frame(root)
        self.algorithms_frame.pack()
        
        # Buttons to run different ML algorithms (disabled until data is preprocessed)
        self.logreg_button = tk.Button(self.algorithms_frame, text="Logistic Regression", command=lambda: self.run_ml("Logistic Regression"), state=tk.DISABLED)
        self.logreg_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.dtree_button = tk.Button(self.algorithms_frame, text="Decision Tree", command=lambda: self.run_ml("Decision Tree"), state=tk.DISABLED)
        self.dtree_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.rf_button = tk.Button(self.algorithms_frame, text="Random Forest", command=lambda: self.run_ml("Random Forest"), state=tk.DISABLED)
        self.rf_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.svm_button = tk.Button(self.algorithms_frame, text="SVM", command=lambda: self.run_ml("SVM"), state=tk.DISABLED)
        self.svm_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.knn_button = tk.Button(self.algorithms_frame, text="KNN", command=lambda: self.run_ml("KNN"), state=tk.DISABLED)
        self.knn_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.nb_button = tk.Button(self.algorithms_frame, text="Naive Bayes", command=lambda: self.run_ml("Naive Bayes"), state=tk.DISABLED)
        self.nb_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Text widget to display results
        self.result_text = tk.Text(root, height=10, width=80)
        self.result_text.pack()
        
        # Figure and canvas to display plots
        self.figure = plt.Figure(figsize=(10, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()
        
        # Variables to hold data and model training/testing splits
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.target_column = None
    
    def load_file(self):
        # Load the CSV file and verify the target column
        file_path = filedialog.askopenfilename()
        if file_path:
            self.df = pd.read_csv(file_path)
            self.target_column = simpledialog.askstring("Input", "Enter the name of the target column:")
            if self.target_column in self.df.columns:
                self.preprocess_button.config(state=tk.NORMAL)
                self.eda_button.config(state=tk.NORMAL)
                messagebox.showinfo("File Loaded", "CSV file has been loaded successfully.")
            else:
                messagebox.showerror("Error", "Target column not found in the CSV file.")
    
    def preprocess_data(self):
        # Preprocess the data: handle missing values, encode labels, split data, apply SMOTE, and scale features
        if self.df is not None and self.target_column in self.df.columns:
            for column in self.df.select_dtypes(include=['number']).columns:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            for column in self.df.select_dtypes(include=['object']).columns:
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            
            label_encoders = {}
            for column in self.df.select_dtypes(include=['object']).columns:
                label_encoders[column] = LabelEncoder()
                self.df[column] = label_encoders[column].fit_transform(self.df[column])
            
            X = self.df.drop(self.target_column, axis=1)
            y = self.df[self.target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Apply SMOTE for oversampling (if needed)
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
            # Enable algorithm buttons after preprocessing
            self.logreg_button.config(state=tk.NORMAL)
            self.dtree_button.config(state=tk.NORMAL)
            self.rf_button.config(state=tk.NORMAL)
            self.svm_button.config(state=tk.NORMAL)
            self.knn_button.config(state=tk.NORMAL)
            self.nb_button.config(state=tk.NORMAL)
            
            messagebox.showinfo("Preprocessing Done", "Data preprocessing is completed.")
        else:
            messagebox.showerror("Error", "No data to preprocess.")
    
    def perform_eda(self):
        # Perform Exploratory Data Analysis (EDA)
        if self.df is not None:
            stats = self.df.describe().to_string()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Basic Statistics:\n")
            self.result_text.insert(tk.END, stats + "\n\n")
            
            self.figure.clear()
            num_columns = self.df.select_dtypes(include=['number']).columns
            num_plots = len(num_columns)
            ncols = 2
            nrows = math.ceil(num_plots / ncols)
            
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
            axes = axes.flatten()
            
            for i, col in enumerate(num_columns):
                self.df[col].hist(bins=20, ax=axes[i])
                axes[i].set_title(col)
            
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            self.canvas.draw()
        else:
            messagebox.showerror("Error", "No data to analyze.")
    
    def evaluate_model(self, model):
        # Evaluate the model using accuracy, recall, and specificity
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred, average='macro')
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        return accuracy, recall, specificity
    
    def run_ml(self, model_name):
        # Train and evaluate the selected machine learning model
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
        
        model = models[model_name]
        model.fit(self.X_train, self.y_train)
        accuracy, recall, specificity = self.evaluate_model(model)
        
        self.display_results(model_name, accuracy, recall, specificity)
        self.plot_results(model_name, accuracy, recall, specificity)
    
    def display_results(self, model_name, accuracy, recall, specificity):
        # Display the results in the text widget
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"{model_name}:\n")
        self.result_text.insert(tk.END, f"  Accuracy: {accuracy:.2f}\n")
        self.result_text.insert(tk.END, f"  Recall: {recall:.2f}\n")
        self.result_text.insert(tk.END, f"  Specificity: {specificity:.2f}\n\n")
    
    def plot_results(self, model_name, accuracy, recall, specificity):
        # Plot the results as a bar chart
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        metrics = [accuracy, recall, specificity]
        metric_names = ['Accuracy', 'Recall', 'Specificity']
        colors = ['green', 'blue', 'orange']  # Define colors for each metric
        
        ax.bar(metric_names, metrics, color=colors)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Scores')
        ax.set_title(f'{model_name} Performance')
        
        self.canvas.draw()

if __name__ == "__main__":
    # Run the application
    root = tk.Tk()
    app = Cardiac_arrest(root)
    root.mainloop()
