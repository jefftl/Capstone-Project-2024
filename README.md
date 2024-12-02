# Plant Classification using Convolutional Neural Networks
- Welcome to the repositiry for predicting 30 different plant species using convalutional neural network architectures. 

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
This project implements and compares three popular deep learning models (ResNet18, VGG16, and AlexNet) for image classification. These models are trained to automatically recognize and categorize images into different classes. Think of it like teaching a computer to look at a picture and tell you what it sees, similar to how humans can instantly recognize objects in photos. The project focuses on making these models as accurate as possible while comparing their different strengths and weaknesses.

## DATA
The data should be organized into three main directories:
- Training set: Used to train the models
- Validation set: Used to tune the models during training
- Test set: Used for final evaluation

Each directory should contain subdirectories, where each subdirectory represents a class and contains images belonging to that class. The data structure should be:
```bash
data/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
    val/
        [same structure as train]
    test/
        [same structure as train]
```
## MODEL 
Three classical CNN architectures are implemented:

ResNet18: A relatively lightweight model that uses residual connections to combat the vanishing gradient problem. Chosen for its excellent performance-to-complexity ratio.
VGG16: A deeper model with a simple, consistent architecture. Selected for its proven track record in image classification tasks.
AlexNet: A pioneering CNN architecture that's simpler than modern alternatives. Included for its historical significance and faster training time.

All models are implemented with batch normalization and modern training techniques like mixed precision training to improve performance and stability.

## HYPERPARAMETER OPTIMSATION
Key hyperparameters include:

Learning rate: Starting at 0.01 with cosine annealing scheduler
Batch size: 32/64 based on GPU memory constraints
Weight decay: 5e-4 for regularization
Momentum: 0.9 for SGD optimizer
Number of epochs: 50
Data augmentation parameters:

Random crop size: 224x224
Color jitter: brightness=0.2, contrast=0.2, saturation=0.2

These values were chosen based on common practices in the literature and empirical testing. The cosine annealing scheduler helps find optimal learning rates throughout training.

## RESULTS
The models are evaluated on three key metrics:

Overall accuracy
Per-class accuracy
Confusion matrix analysis

Each model produces:

Training/validation loss curves
Validation accuracy progression
Training time per epoch
Detailed classification report
Confusion matrix visualization

Example visualization code is implemented to track:

Loss curves
Accuracy progression
Per-class performance
Training efficiency

The confusion matrix provides insights into:

Which classes are easily distinguished
Where misclassifications commonly occur
Overall model reliability

[Note: Specific accuracy numbers and plots would be included here based on your actual results]

## CONTACT DETAILS
- Jeffrey Lewis
- jeffreytudorlewis@gmail.com
