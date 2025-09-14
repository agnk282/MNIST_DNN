# MNIST MLP Classifier

This project trains a Multi-Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset using TensorFlow/Keras.

## Features
- Loads and visualizes MNIST data
- Preprocesses images and labels
- Builds a robust MLP with dropout regularization
- Uses early stopping and model checkpointing
- Plots training/validation accuracy and loss
- Evaluates with accuracy, loss, classification report, and confusion matrix
- Saves the trained model for future use

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
Run the main script:
```bash
python main.py
```

## Output
- Training and validation plots
- Test accuracy and loss
- Classification report and confusion matrix
- Saved models: `best_mnist_mlp.h5`, `final_mnist_mlp.h5`

## Customization
You can adjust model architecture, training parameters, or add more advanced features in `main.py`.

