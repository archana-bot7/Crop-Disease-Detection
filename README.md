# 🌱 Crop Doctor – AI-Based Crop Disease Detection

An AI-powered crop disease detection application built using Deep Learning and Computer Vision to identify plant diseases from leaf images. Crop Doctor uses the **EfficientNet-B0 Convolutional Neural Network (CNN)** architecture to classify crop leaf images and help users recognize potential plant diseases quickly.

The project aims to make preliminary crop disease identification more accessible by allowing users to upload an image of an affected plant leaf and receive an AI-generated prediction through a simple, interactive web application.

---

## 📌 1. Project Overview

Agriculture is one of the most important sectors in many developing countries. However, crop diseases remain a major challenge for farmers, often resulting in reduced crop production, financial losses, and food insecurity.

Traditionally, identifying crop diseases requires farmers to recognize visible symptoms or consult agricultural experts. This process can sometimes be time-consuming, especially when expert assistance is not readily available.

**Crop Doctor addresses this problem using Artificial Intelligence and Deep Learning.**

The application analyzes uploaded plant leaf images using a trained deep learning model and predicts the corresponding disease category based on patterns learned during training.

The goal is to provide an accessible, technology-driven approach to preliminary crop disease identification.

## 🎯 2. Project Objectives

- Developed a deep learning model capable of classifying crop diseases from leaf images.
- Implemented EfficientNet-B0 for image classification.
- Prepared and preprocess image datasets for model training.
- Evaluated model performance using training and validation accuracy.
- Developed an interactive web application using Gradio.
- Connected the trained deep learning model with the application for real-time image prediction.
- Createed a simple interface that makes AI-based disease identification accessible to non-technical users.

## ⚙️ 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning model development and training |
| EfficientNet-B0 | CNN architecture for image classification |
| NumPy | Numerical operations |
| Pandas | Dataset organization and analysis |
| Matplotlib | Training visualization |
| Pillow (PIL) | Image processing |
| Gradio | Interactive web application |
| Jupyter Notebook | Model experimentation and development |
| Git & GitHub | Version control and project management |

## 🧠 4. Deep Learning Model – EfficientNet-B0

Crop Doctor uses **EfficientNet-B0**, a convolutional neural network architecture designed to achieve a good balance between computational efficiency and classification performance.

Unlike traditional image classification approaches, EfficientNet uses a compound scaling method that systematically balances network depth, width, and image resolution.

This makes it suitable for image classification tasks where both accuracy and computational efficiency are important.

### Model Development Process

**Step 1: Dataset Preparation**

The image dataset is organized into appropriate categories for supervised image classification. Each image is associated with its corresponding crop disease label.

**Step 2: Image Preprocessing**

Images are prepared in a consistent format and resolution before being passed into the neural network. This helps the model learn meaningful visual patterns.

**Step 3: Model Training**

The EfficientNet-B0 architecture is trained using the prepared dataset to recognize visual characteristics associated with different crop disease categories.

**Step 4: Model Validation**

The dataset is divided into training and validation subsets to evaluate how well the model performs on images it has not seen during training.

**Step 5: Model Saving**

The trained model is saved for future use in the Gradio application. Checkpoint functionality is also implemented to support training recovery and continuation.

## 📊 5. Model Performance

The model achieved the following performance during training and validation:

| Evaluation Metric | Accuracy |
|---|---:|
| Training Accuracy | 94.63% |
| Validation Accuracy | 93.05% |

The validation accuracy of 93.05% indicates that the model learned useful image patterns and demonstrated promising classification performance on the validation dataset.

These results represent the model's performance on the dataset used during development and should not be interpreted as a guarantee of accuracy under all real-world agricultural conditions.

## 🖥️ 6. Application – Gradio Interface

The trained model is integrated into an interactive web application developed using Gradio.

The application provides a user-friendly interface through which users can submit plant leaf images for disease classification.

### Application Workflow

1. The user opens the Crop Doctor web application.
2. The user uploads an image of a plant leaf.
3. The application processes the uploaded image.
4. The processed image is passed to the trained EfficientNet-B0 model.
5. The model analyzes the image and generates a disease classification prediction.
6. The application displays the prediction to the user.

### Key Application Components

- Image upload functionality.
- Image preprocessing before inference.
- Integration of the trained EfficientNet-B0 model.
- Automated image classification.
- Interactive Gradio-based interface.

## 🔄 7. Project Workflow

The complete project follows an end-to-end machine learning workflow:

```text
          IMAGE DATASET
                |
                ▼
       DATASET PREPARATION
                |
                ▼
       IMAGE PREPROCESSING
                |
                ▼
       EFFICIENTNET-B0 MODEL
                |
                ▼
          MODEL TRAINING
                |
                ▼
         MODEL VALIDATION
                |
                ▼
          MODEL SAVING
                |
                ▼
        GRADIO APPLICATION
                |
                ▼
         IMAGE UPLOAD
                |
                ▼
        DISEASE PREDICTION
                |
                ▼
        DISPLAY RESULTS
```

Potential applications include:

- Helping farmers perform preliminary crop disease identification.
- Supporting agricultural students and researchers in understanding plant disease classification.
- Providing accessible AI-based image classification through a web interface.
- Demonstrating the potential of computer vision in precision agriculture.
- Supporting the development of smarter agricultural technology solutions.

**Important:** Crop Doctor is an experimental disease classification tool, not a replacement for professional agricultural diagnosis or treatment recommendations.



