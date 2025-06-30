# Face Recognition Project

This project implements a basic face recognition system using Python, OpenCV, and scikit-learn. It leverages Principal Component Analysis (PCA) for dimensionality reduction (often referred to as "Eigenfaces") and a Support Vector Machine (SVM) for classification.

## Table of Contents

* [About the Project](#about-the-project)

* [Features](#features)

* [Technologies Used](#technologies-used)

* [Dataset](#dataset)

* [Setup and Installation](#setup-and-installation)

* [Usage](#usage)

* [Results](#results)

* [Contributing](#contributing)

* [License](#license)

## About the Project

This project demonstrates a classic approach to face recognition. It involves:

1. **Loading and Preprocessing**: Reading grayscale images from a specified directory and resizing them to a uniform size.

2. **Feature Extraction with PCA**: Applying PCA to reduce the dimensionality of the image data, transforming high-dimensional pixel data into a lower-dimensional feature space (Eigenfaces).

3. **Model Training**: Training a Support Vector Machine (SVM) classifier on the PCA-transformed training data.

4. **Prediction and Evaluation**: Evaluating the trained model's performance on unseen test data using accuracy, classification report, and confusion matrix.

5. **Visualization**: Displaying sample test images along with their predicted and true labels.

## Features

* Image loading and resizing.

* PCA for dimensionality reduction (Eigenfaces).

* SVM classifier for face recognition.

* Splitting data into training and testing sets.

* Evaluation metrics: Accuracy, Classification Report, Confusion Matrix.

* Visualization of predictions.

## Technologies Used

* Python

* `numpy`: For numerical operations, especially array manipulation.

* `opencv-python` (`cv2`): For image loading, preprocessing (resizing, grayscale conversion).

* `scikit-learn`:

  * `PCA`: For Principal Component Analysis.

  * `SVC`: For Support Vector Machine classification.

  * `train_test_split`: For splitting data.

  * `classification_report`, `confusion_matrix`, `accuracy_score`: For model evaluation.

* `matplotlib`: For plotting and visualization.

## Dataset

The project is designed to work with a dataset of facial images. The provided Jupyter notebook assumes the dataset is located at: `"drive/MyDrive/ORL"`. This suggests the project was initially set up in a Google Colab environment with Google Drive mounted.

The `load_images_from_folder` function handles:

* Iterating through subdirectories, where each subdirectory name is expected to represent a unique person (e.g., `s1`, `s2`, `s3`, etc.).

* Loading images as grayscale.

* Resizing images to a target size of `(92, 112)`.

**Note**: You will need to provide your own ORL (Olivetti Research Laboratory) face dataset or a similar structured dataset in the specified path for the code to run successfully.

## Setup and Installation

1. **Clone the Repository (if applicable)**:

git clone 


2. **Install Dependencies**:
It is recommended to use a virtual environment.

pip install numpy opencv-python scikit-learn matplotlib


3. **Dataset Preparation**:

* Obtain the ORL Face Database or another suitable face dataset.

* Organize your dataset such that each person's images are in a separate subdirectory, and the subdirectory names can be parsed into integer labels (e.g., `s1`, `s2`, ...).

* Update the `dataset_folder` variable in the script to point to the root directory of your dataset (e.g., `"path/to/your/ORL"`). If you are using Google Colab, the path `"drive/MyDrive/ORL"` is typical for a dataset stored on Google Drive.

## Usage

To run the face recognition script:

1. **Open the Jupyter Notebook**:
If you're using a `.ipynb` file (like `Face Recognisation project.ipynb`), open it with Jupyter Notebook or Google Colab.

2. **Execute Cells**:
Run each cell sequentially. The notebook performs the following steps:

* Imports necessary libraries.

* Loads images and labels from the specified `dataset_folder`.

* Splits the data into training and testing sets (80% training, 20% testing).

* Reshapes the image data for PCA.

* Applies PCA to reduce dimensionality to `n=80` components.

* Trains an SVC classifier on the PCA-transformed training data.

* Makes predictions on the test set.

* Prints the accuracy score.

* Prints a detailed classification report (precision, recall, f1-score).

* Prints the confusion matrix.

* Displays 10 sample test images with their predicted and true labels.

## Results

The provided notebook output indicates:

* **Number of images loaded**: 400

* **Accuracy**: 0.9625 (96.25%)

The classification report and confusion matrix provide more detailed insights into the model's performance per class. The visualization step shows how the model predicts labels for a few test images compared to their true labels.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests to improve the project.

## License

\[Specify your license here, e.g., MIT, Apache 2.0, etc.\]
