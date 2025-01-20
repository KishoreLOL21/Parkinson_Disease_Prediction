🧠 Parkinson's Disease Detection using MRI Scans 🏥
Welcome to the Parkinson's Disease Detection repository! This project focuses on building a hybrid machine learning model to detect Parkinson's disease using MRI scans. The model combines classical Convolutional Neural Networks (CNNs) with a quantum-inspired layer for accurate binary classification (Non-PD vs. PD).

🌟 Key Features
Hybrid Model: Combines classical CNNs with a quantum-inspired layer for improved performance.

MRI Data Preprocessing: Converts, resizes, and normalizes MRI images for model input.

Comprehensive Evaluation: Includes ROC curves, precision-recall curves, and class-wise accuracy.

Visualizations: Activation maps, weight heatmaps, and 3D loss landscapes for better understanding.

Easy to Use: Well-documented code and modular design for easy customization.

🛠️ Technologies Used
Python 🐍

PyTorch 🔥

NumPy 🧮

Matplotlib 📊

Seaborn 🎨

Scikit-learn 🤖

📂 Dataset
The dataset used in this project is the NTUA Parkinson Dataset, which contains MRI scans of patients categorized into:

Non-PD Patients: Healthy individuals without Parkinson's disease.

PD Patients: Individuals diagnosed with Parkinson's disease.

The dataset is organized into folders for each patient, with subfolders containing MRI images in PNG format.

🚀 How It Works
1. Data Preprocessing
Convert MRI images to grayscale.

Resize images to 96x96 pixels.

Normalize pixel values to the range [0, 1].

Split the dataset into training (70%), validation (15%), and test (15%) sets.

2. Model Architecture
Classical CNN: Two convolutional layers with max-pooling and adaptive average pooling.

Quantum-like Layer: A fully connected layer inspired by quantum computing principles.

Hybrid Model: Combines the classical CNN and quantum-like layer for binary classification.

3. Training
Train the model using the Adam optimizer and CrossEntropyLoss.

Track training and validation losses.

Save the best model based on validation loss.

4. Evaluation
Validation Accuracy: Measures overall model performance.

ROC Curve: Evaluates the model's ability to distinguish between Non-PD and PD classes.

Precision-Recall Curve: Highlights the trade-off between precision and recall.

Class-wise Accuracy: Provides insights into model performance for each class.

5. Visualization
Activation Maps: Visualize what the model is learning.

Weight Heatmaps: Inspect the learned filters.

3D Loss Landscape: Illustrate the optimization process.

📊 Results
Validation Accuracy: Achieves X% accuracy on the validation set.

ROC Curve: Area under the curve (AUC) of Y.

Precision-Recall Curve: Average precision (AP) of Z.

Class-wise Accuracy:

Non-PD: A%

PD: B%

🖼️ Visualizations
Activation Map
Activation Map
Visualization of the activation map for the first convolutional layer.

Weight Heatmap
Weight Heatmap
Heatmap of weights for the first filter in the first convolutional layer.

3D Loss Landscape
3D Loss Landscape
Synthetic 3D plot of the loss landscape.

🛠️ Installation
Clone the repository:
git clone https://github.com/your-username/parkinsons-detection.git
cd parkinsons-detection
Install the required dependencies:
pip install -r requirements.txt
Download the dataset from Kaggle and place it in the data folder.

🙏 Acknowledgments
NTUA Parkinson Dataset: Provided by Kaggle.

PyTorch Community: For providing an excellent deep learning framework.

Open Source Contributors: For their valuable tools and libraries.

