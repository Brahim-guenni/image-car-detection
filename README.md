# Car Plate Detection and Recognition System

This project implements a complete pipeline for detecting car license plates from images, extracting the text using Tesseract OCR, and verifying the digits using Deep Learning (CNNs) trained on the MNIST dataset.

## 🚀 Project Overview

The system operates in three main stages:
1.  **Detection**: Locates the car license plate using OpenCV and Haar Cascades.
2.  **OCR Extraction**: Extracts the text from the plate using Tesseract OCR.
3.  **Verification**: Uses a Convolutional Neural Network (CNN) to independently recognize and verify the digits on the plate, providing a confidence score.

## 🛠️ Requirements & Installation



## 🧠 Models Used & Conclusions

To ensure the best accuracy for digit recognition, the project trains and compares three different neural network architectures on the MNIST dataset.

### 1. Simple CNN
- **Architecture**: 3 Convolutional layers (32, 64, 64 filters) followed by Max Pooling.
- **Purpose**: To capture spatial features like edges and curves essential for digit recognition.
- **Performance**: Good baseline accuracy but may struggle with complex variations.

### 2. Logistic Regression Classifier


- ** Architecture: A linear classification model that operates on flattened image inputs. Each image is converted into a one-dimensional feature vector and passed through a softmax function to predict class probabilities.

- ** Purpose: Serves as a simple and interpretable baseline model for character recognition. Logistic Regression allows evaluation of how well basic linear decision boundaries perform compared to more complex neural networks.

- ** Performance**: Achieves lower accuracy than convolutional neural networks due to its inability to capture spatial features and local patterns in images, but offers fast training time and stable performance on clean, well-aligned data.

## 📊 Key Conclusions

- **CNN Superiority**: The Convolutional Neural Network (CNN) demonstrates superior performance compared to Logistic Regression due to its ability to automatically learn spatial features such as edges, curves, and structural patterns within characters.

- **Role of Logistic Regression**: Logistic Regression serves as a strong baseline model. While it offers fast training and simple interpretation, its performance is limited because flattening the image removes important spatial information.

- **Evaluation Metrics**: Model performance is evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score**. Comparative results are presented through tables and visualizations such as confusion matrices and training curves, providing a transparent and objective comparison between models.

## 📂 Project Structure

- **`Car Plate Detection with OpenCV and TesseractOCR.ipynb`**: The main Jupyter Notebook containing all code.
- **`images/`**: Folder containing sample car images.
- **`haar_cascades/`**: Folder containing the `haarcascade_russian_plate_number.xml` file used for detection.

## 📝 Usage Guide

1.  **Open the Notebook**: Launch Jupyter Notebook and open `Car Plate Detection with OpenCV and TesseractOCR.ipynb`.
2.  **Run Cells**: Execute the cells sequentially.
    - **Step 1**: Installs dependencies.
    - **Step 2**: Detects the license plate in the sample image (`image2.png`).
    - **Step 3**: Preprocesses the image (Grayscale, Blur) and runs Tesseract OCR.
    - **Step 4**: Trains the 2 MNIST models and displays the comparison metrics.
    - **Step 5**: Segments the license plate characters using OpenCV contours.
    - **Step 6**: Predicts each segmented digit using the best CNN model and displays the result with confidence percentages.

## 📚 Concepts & Code Explanation

### 🔑 Key Concepts Defined

- **Haar Cascade Classifier**:
  - *What is it?* A classic machine learning object detection method proposed by Paul Viola and Michael Jones.
  - *Why use it?* It is fast and efficient for detecting rigid objects like faces or, in our case, **license plates**. It uses a "cascade" of simple features (edge, line, rectangle) to quickly reject non-object regions.

- **CNN (Convolutional Neural Network)**:
  - *What is it?* A class of deep neural networks, most commonly applied to analyzing visual imagery.
  - *Why use it?* Unlike traditional algorithms, CNNs can automatically learn to recognize patterns (edges -> shapes -> objects). We use it to **recognize the digits (0-9)** on the license plate with high accuracy, trained on the famous MNIST dataset.

- **Thresholding (Otsu's Method)**:
  - *What is it?* A technique to convert a grayscale image into a binary image (pure black and white).
  - *Why use it?* It simplifies the image so the computer can easily separate the "foreground" (the digits) from the "background". **Otsu's method** automatically finds the optimal threshold value to minimize variance between the two classes.

- **Contours**:
  - *What is it?* Curves joining all the continuous points along a boundary having the same color or intensity.
  - *Why use it?* We use contour detection to **segment** (isolate) each individual digit on the license plate so we can feed them one by one into our CNN model.

### 💻 Function Breakdown

- **`carplate_detect(image)`**
  - Uses the `haarcascade_russian_plate_number.xml` file to scan the image.
  - Draws a rectangle around the detected plate for visualization.

- **`carplate_extract(image)`**
  - Similar to detection, but instead of drawing a box, it **crops and returns** the specific region of the image containing the license plate.
  - This cropped image is what gets passed to the OCR and CNN.

- **`enlarge_img(image, scale_percent)`**
  - Resizes the image to a larger scale using interpolation.
  - *Why?* Small, low-resolution text is difficult for Tesseract and our segmentation logic to process. Enlarging it improves accuracy.

- **`create_simple_cnn()` / `create_deep_cnn_dropout()` / `create_mlp()`**
  - These helper functions define the structure (layers) of the neural networks.
  - Using functions allows us to easily instantiate multiple fresh models for comparison.

- **`extract_and_predict_characters(image, model)`**
  - **The "Brain" of the verification step.**
  - 1. **Preprocesses**: Applies inverse binary thresholding (white text on black background).
  - 2. **Segments**: Finds contours to locate potential digits.
  - 3. **Filters**: Discards noise (contours that are too small to be digits).
  - 4. **Normalizes**: Resizes each digit to 28x28 pixels (matching the MNIST training data format).
  - 5. **Predicts**: Feeds each processed digit into the trained Best Model to get the number.

## 🔍 Key Technologies

- **OpenCV (`cv2`)**: Image processing, contour detection, and Haar Cascade classification.
- **Tesseract OCR**: Optical Character Recognition engine.
- **TensorFlow/Keras**: Building and training Deep Learning models.
- **LogisticRegression
- **Pandas**: Data manipulation for model comparison.
- **Matplotlib**: Visualization of images and results.


accuracy 
f1 score



