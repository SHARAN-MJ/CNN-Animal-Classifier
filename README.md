# CNN-Animal-Classifier

This repository contains a Jupyter Notebook (`cnn.ipynb`) demonstrating a **Convolutional Neural Network (CNN)** built with PyTorch to classify images into three categories: **cats**, **dogs**, and **pandas**.
<img width="2816" height="1536" alt="Gemini_Generated_Image_6x6ir06x6ir06x6i" src="https://github.com/user-attachments/assets/923caa30-b808-42ee-9264-21e2a6b95d42" />

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The goal of this project is to implement and train a CNN that can accurately distinguish between cats, dogs, and pandas.  
The notebook demonstrates:
- Loading and preprocessing image data using `torchvision.datasets.ImageFolder` and transforms.
- Building a custom CNN architecture using `torch.nn.Module`.
- Training the model and tracking performance over epochs.
- Evaluating the model with **loss & accuracy plots** and a **confusion matrix**.

This project is beginner-friendly yet covers all the important steps of building an image classifier in PyTorch.

---

## Dataset

The dataset should be structured like this:

<img width="2816" height="1536" alt="Gemini_Generated_Image_st5j9vst5j9vst5j" src="https://github.com/user-attachments/assets/e6f1f916-693c-44b8-9de8-b4c7d9f2e977" />


dataset/
└── Project/
├── train/
│ ├── cats/
│ ├── dogs/
│ └── panda/
└── test/
├── cats/
├── dogs/
└── panda/


> 📌 **Important:**  
> The notebook assumes the dataset path is `/content/drive/My Drive/dataset/Project` (for Google Colab).  
> If you use a different environment or local machine, change the `root` variable in `cnn.ipynb` to your dataset path (e.g., `./dataset/Project`).

---

## Model Architecture

The custom CNN (`CNN` class) includes:
- Two convolutional layers (`nn.Conv2d`) each followed by ReLU activation and max pooling (`nn.MaxPool2d`).
- First conv layer: 16 output channels, second conv layer: 32 output channels.
- A fully connected layer (`nn.Linear`) reducing features to 128.
- Final output layer mapping 128 features to 3 classes.

Input images are resized to **224×224**. After the conv and pooling layers, the flattened feature size becomes `32 * 56 * 56`, which is passed to the fully connected layers.

---

## Usage

1. Prepare your dataset as described in the [## Dataset](#dataset) section.

2. Open the notebook:

```bash
jupyter notebook cnn.ipynb

3. Run each cell step by step:

Mount Google Drive (if using Colab)

Import libraries and define transforms

Load data

Build and train the model

Visualize training results

## Results

### 📉 Loss & Accuracy over Epochs

Shows how the model improves during training.

<img width="1001" height="470" alt="download" src="https://github.com/user-attachments/assets/9d2c21c7-5d32-490a-8be8-66240f683206" />

---

### ✅ Confusion Matrix

Displays actual vs predicted labels.

<img width="542" height="482" alt="download (1)" src="https://github.com/user-attachments/assets/4d6af5a5-6763-4866-a339-1c35cbb7cddc" />

---

## Contributing

Feel free to fork this repository, make improvements, and open pull requests!  
All contributions to make this project better are welcome.

---

## License

This project is licensed under the MIT License.




