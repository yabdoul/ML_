# MNIST Digit Recognition with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It includes model training, evaluation, and inference on custom images.

## Project Structure

```
ML_/
├── data/                # (Optional) Data directory
├── MNIST/               # MNIST data (raw)
├── src/
│   ├── model.py         # Model definition and training script
│   ├── test.py          # Script to test the model on a custom image
│   ├── mnist_model.pth  # Trained model weights
│   └── image.png        # Example image for inference
├── README.md            # Project documentation
└── ...
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow

Install dependencies:

```bash
pip install torch torchvision pillow
```

## Training the Model

The model is defined and trained in `src/model.py`. To train the model on the MNIST dataset and save the weights:

```bash
cd src
python model.py
```
This will download the MNIST dataset (if not present), train the CNN, and save the model as `mnist_model.pth`.

## Testing/Inference

To predict the digit in a custom image (`image.png`):

1. Place a 28x28 grayscale image of a digit as `src/image.png` (or modify the path in `test.py`).
2. Run:

```bash
cd src
python test.py
```
The script will output the predicted digit.

## Model Architecture

- 1 convolutional layer (32 filters, 3x3 kernel)
- ReLU activations
- Max pooling
- Fully connected layer (output: 10 classes)

## Notes

- The MNIST dataset is automatically downloaded to the `MNIST/raw/` directory.
- The model expects grayscale images of size 28x28 for inference.

## License

MIT License