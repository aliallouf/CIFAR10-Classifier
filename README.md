# CIFAR-10 Image Classifier

This project implements an image classification model trained on the CIFAR-10 dataset using Transfer Learning with InceptionV3. It can classify images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Project Process
1. **Data Preparation**: Loaded the CIFAR-10 dataset (60,000 images) and normalized pixels to a 0-1 range.
2. **Model Architecture**:
   - Used a pre-trained **InceptionV3** model (without the top layers).
   - Added a Global Average Pooling layer, a Dropout layer (0.2), and a final Dense layer with a **Softmax** activation for 10-class prediction.
3. **Training**:
   - The model was trained with an early stopping callback that triggers once 90% accuracy is reached.
   - Final Training Accuracy: ~91.68%
   - Final Validation Accuracy: ~81.45%
4. **Inference**: A custom script loads the `.keras` model and performs batch predictions on local image files.

## How to Use

### Installation
```bash
pip install tensorflow numpy matplotlib