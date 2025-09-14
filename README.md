# MNIST Classification: MLP, CNN, and Transformer Models

This project demonstrates three approaches to classifying handwritten digits from the MNIST dataset using TensorFlow/Keras:

## 1. Multi-Layer Perceptron (MLP)
- **File:** `main.py`
- **Description:** A fully connected neural network that flattens the 28x28 images and passes them through dense layers with dropout for regularization.
- **Features:**
  - Dropout regularization
  - Early stopping and model checkpointing
  - Training/validation accuracy and loss plots
  - Classification report and confusion matrix
  - Model saved as `final_mnist_mlp.h5`

## 2. Convolutional Neural Network (CNN)
- **File:** `cnn.py`
- **Description:** A deep learning model that uses convolutional and pooling layers to extract spatial features from images, followed by dense layers for classification.
- **Features:**
  - Two convolutional layers and max pooling
  - Dropout regularization
  - Early stopping and model checkpointing
  - Training/validation accuracy and loss plots
  - Classification report and confusion matrix
  - Model saved as `final_mnist_cnn.h5`

## 3. Vision Transformer (ViT)
- **File:** `transformer.py`
- **Description:** A transformer-based model that splits each image into patches, embeds them, adds positional encoding, and processes them with transformer encoder layers before classification.
- **Features:**
  - Patch extraction and embedding
  - Multi-head self-attention and MLP blocks
  - Early stopping and model checkpointing
  - Training/validation accuracy and loss plots
  - Classification report and confusion matrix
  - Model saved as `final_mnist_transformer.h5`

## Requirements
- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- seaborn

Install dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

## Usage
Run the desired script:
```bash
python main.py        # MLP
python cnn.py         # CNN
python transformer.py # Vision Transformer
```

## Output
- Training and validation plots
- Test accuracy and loss
- Classification report and confusion matrix
- Saved models: `best_mnist_mlp.h5`, `final_mnist_mlp.h5`, `best_mnist_cnn.h5`, `final_mnist_cnn.h5`, `best_mnist_transformer.h5`, `final_mnist_transformer.h5`

## Customization
You can adjust model architectures, training parameters, or add more advanced features in each script.

## License
This project is for educational purposes.
