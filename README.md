# CIFAR-10 Object Recognition using ResNet50

A deep learning project implementing object recognition on the CIFAR-10 dataset using transfer learning with ResNet50 architecture.

## Project Overview

This project demonstrates the use of transfer learning with a pre-trained ResNet50 model to classify images from the CIFAR-10 dataset. The implementation compares a simple neural network baseline with a more sophisticated ResNet50-based model.

## Dataset

**CIFAR-10** contains 60,000 32x32 color images across 10 classes:
- Airplane
- Automobile  
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

**Dataset Split:**
- Training: 50,000 images
- Testing: 10,000 images
- Each class contains 5,000 training images

## Implementation Details

### Data Preprocessing

1. **Image Loading**: Images are loaded from PNG files and converted to NumPy arrays
2. **Label Encoding**: Text labels are mapped to numerical values (0-9)
3. **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
4. **Train-Test Split**: 80-20 split with validation subset

### Model Architectures

#### Baseline Model (Simple Neural Network)
```
Input (32x32x3) → Flatten → Dense(64, relu) → Dense(10, softmax)
```
- **Test Accuracy**: ~40.8%
- Simple architecture for comparison baseline

#### ResNet50 Transfer Learning Model
```
Input (32x32x3) → UpSampling(x8 to 256x256) → ResNet50(pretrained) → 
Flatten → BatchNorm → Dense(128, relu) → Dropout(0.5) → 
BatchNorm → Dense(64, relu) → Dropout(0.5) → 
BatchNorm → Dense(10, softmax)
```

**Key Features:**
- Pre-trained ResNet50 on ImageNet (weights='imagenet')
- Upsampling layers to meet ResNet50 input requirements (256x256)
- Batch normalization for training stability
- Dropout layers (0.5) to prevent overfitting
- RMSprop optimizer (lr=2e-5)

## Results

### ResNet50 Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 97.6% | 94.6% | 93.9% |
| **Loss** | 0.163 | 0.202 | 0.232 |

### Training Progress (10 Epochs)

**Accuracy Improvement:**
- Epoch 1: 46.3% → Epoch 10: 97.6% (train)
- Epoch 1: 76.7% → Epoch 10: 94.6% (validation)

**Loss Reduction:**
- Epoch 1: 1.640 → Epoch 10: 0.163 (train)
- Epoch 1: 0.822 → Epoch 10: 0.202 (validation)

## Technical Stack

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **PIL**: Image processing
- **Matplotlib**: Visualization
- **scikit-learn**: Train-test splitting

## Key Techniques

1. **Transfer Learning**: Leveraging pre-trained ResNet50 weights
2. **Image Upsampling**: Adapting 32x32 images to ResNet50's 256x256 input
3. **Regularization**: Dropout and batch normalization
4. **Data Normalization**: Pixel scaling for better convergence

## Installation & Usage

```bash
# Install dependencies
pip install tensorflow numpy pandas pillow matplotlib scikit-learn kaggle py7zr

# Configure Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c cifar-10

# Extract dataset
py7zr train.7z
```

## Model Training

```python
# Compile model
model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

# Train model
history = model.fit(
    X_train_scaled, 
    Y_train,
    validation_split=0.1,
    epochs=10
)
```

## Evaluation

```python
# Evaluate on test set
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print(f'Test Accuracy: {accuracy:.4f}')
```

## Visualizations

The project includes training history plots showing:
- Loss curves (training vs validation)
- Accuracy curves (training vs validation)

## Future Improvements

- Data augmentation for improved generalization
- Hyperparameter tuning (learning rate, batch size)
- Fine-tuning ResNet50 layers
- Ensemble methods
- Class activation mapping for interpretability

**Note**: GPU acceleration recommended for training the ResNet50 model.
