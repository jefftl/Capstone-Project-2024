# VGG16 Model Card

## Model Description

**Input:** 
- RGB images transformed as follows:
 Training:
 - Resize to 256x256
 - Random crop to 224x224
 - Random horizontal flip
 - Random rotation (±15 degrees)
 - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
 - ToTensor conversion
 - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

 Evaluation:
 - Resize to 256x256
 - Center crop to 224x224
 - ToTensor conversion
 - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Output:**
- Probability distribution across num_classes
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1

**Model Architecture:**
Features Network (5 blocks):
- Block 1:
 - Conv 3->64, 3x3, pad 1, ReLU
 - Conv 64->64, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 2:
 - Conv 64->128, 3x3, pad 1, ReLU
 - Conv 128->128, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 3:
 - Conv 128->256, 3x3, pad 1, ReLU
 - Conv 256->256, 3x3, pad 1, ReLU
 - Conv 256->256, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 4:
 - Conv 256->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 5:
 - Conv 512->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

Classifier:
- Flatten
- Linear: 512*7*7 -> 4096, ReLU
- Dropout
- Linear: 4096 -> 4096, ReLU
- Dropout
- Linear: 4096 -> num_classes

Initialization:
- Conv layers: Kaiming normal (fan_out, relu)
- Conv biases: Constant(0)
- Linear weights: Normal(0, 0.01)
- Linear biases: Constant(0)

## Performance
- Training approach:
 - Comprehensive data augmentation
 - Dropout for regularization
 - Uniform 3x3 convolutions
 - Weight initialization strategy
 - ReLU activations
- Metrics monitored:
 - Training/validation loss
 - Validation accuracy
 - Per-class accuracy
 - Confusion matrix

## Limitations
- Deep network (16 layers) requires significant computation
- Large number of parameters
- No batch normalization
- Memory intensive, especially for:
 - Large feature maps
 - Large fully connected layers
 - High batch sizes
- Fixed input size requirement (224x224)

## Trade-offs
- Architecture Design:
 - Uniform 3x3 filters are efficient
 - Deep network provides good feature hierarchy
 - But requires significant memory and computation
- Training Considerations:
 - Comprehensive augmentation improves robustness
 - But increases training time
- Memory vs Batch Size:
 - Large model size limits batch size
 - Small batches may affect training stability
- Regularization:
 - Dropout helps prevent overfitting
 - But doubles forward pass memory during training
