# Plant Classification using Convolutional Neural Networks
- Welcome to the repositiry for predicting 30 different plant species using convalutional neural network architectures. 

## Useful Links
- [split_dataset.ps1](https://github.com/jefftl/Capstone-Project-2024/blob/main/src/split-dataset.ps1)
- [Plant Classification Dataset](https://www.kaggle.com/datasets/marquis03/plants-classification/data)
- [Models Already Trained on the Dataset](https://drive.google.com/drive/folders/1PrgbfW7dpJPjd12sRu-GmuXmLbf243fr?usp=sharing)

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

  # Plant Classification using Convolutional Neural Networks
Welcome to the repository for predicting 30 different plant species using convolutional neural network architectures. 

## Useful Links
- [split_dataset.ps1](https://github.com/jefftl/Capstone-Project-2024/blob/main/src/split-dataset.ps1)
- [Plant Classification Dataset](https://www.kaggle.com/datasets/marquis03/plants-classification/data)
- [Models Already Trained on the Dataset](https://drive.google.com/drive/folders/1PrgbfW7dpJPjd12sRu-GmuXmLbf243fr?usp=sharing)

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
This project implements and compares four popular deep learning models (ResNet18, VGG16, AlexNet, and LeNet-5) for plant image classification. These models are trained to recognize and categorize images of 30 different plant species. I tried to build the models in such a way that they could be reused on other datasets. Each model can be run independantly and then compared against one another using [compare_models](https://github.com/jefftl/Capstone-Project-2024/blob/main/src/compare_models.ipynb). The best version of each model is also saved when running, allowing users to reuse the already trained models. 

## Using the project on other datasets. 

- The models require the dataset to be split into the below structure.
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
if the model is not in this structure I have provided [split_dataset.ps1](https://github.com/jefftl/Capstone-Project-2024/blob/main/src/split-dataset.ps1) which allows the user to split the data set randomly creating the file structure needed. The user must also set the percentage split for training, validation and testing. Please see example of how to call the powershell script. 
```powershell
.\split-dataset.ps1 -sourceDir "images" -trainPercent 70 -valPercent 20 -testPercent 10
```
Once run one or all the models can be selected to be run but please ensure you update the following file locations:
```python
train_loc = '...' # Training dataset
val_loc = '...' # Validation dataset
test_loc = '...' # Test dataset
model_path = '...' # The best model that was saved. 
```
This will provide the graphs seen in the model cards. Finally if you have run all the models you can use the [compare_models](https://github.com/jefftl/Capstone-Project-2024/blob/main/src/compare_models.ipynb) file to get an overall summary of how each of the models performed. Please just make sure you update the check point of each of the models. 
```python
'checkpoint': '...' # Location of your best model for each architecture
```

## DATA
The dataset consists of plant images organized into three main directories:
- Training set (70%): Used to train the models
- Validation set (10%): Used to tune the models during training
- Test set (20%): Used for final evaluation

Each directory contains 30 subdirectories (one per plant species) with approximately 200 images per class.

## MODEL COMPARISON
### Accuracy Performance
- ResNet18: 85.4% (Best performer)
- VGG16: 81.5%
- AlexNet: 75.2%
- LeNet-5: 42.8% (Poorest performer)

### Model Sizes
- VGG16: 512.6MB (Largest)
- AlexNet: 217.9MB
- ResNet18: 42.7MB
- LeNet-5: 0.2MB (Smallest)

### Inference Times
- ResNet18: 4.2ms
- VGG16: 2.2ms
- AlexNet: 1.2ms
- LeNet-5: 0.8ms

### Key Findings
- ResNet18 achieves the best accuracy but has the slowest inference time
- LeNet-5 is the fastest and smallest but significantly underperforms
- VGG16 provides good accuracy but requires the most storage
- AlexNet offers a balanced trade-off between performance and resources

## HYPERPARAMETER OPTIMISATION
Common hyperparameters across all models:
- Learning rate: 0.01 with cosine annealing scheduler
- Optimizer: SGD with momentum 0.9
- Weight decay: 5e-4
- Number of epochs: 50
- Batch sizes: 
  - Training: 64
  - Validation/Testing: 32

Data augmentation:
- Random crop to 224x224
- Random horizontal flip
- Color jitter (brightness=0.2, contrast=0.2)

For more information on the hyperparameters and Data augmentation please consult the [model card](https://github.com/jefftl/Capstone-Project-2024/tree/main/documents).

## RESULTS
### Model Performance Summary
| Model    | Accuracy | Size   | Inference Time |
|----------|----------|--------|----------------|
| ResNet18 | 85.4%    | 42.7MB | 4.2ms         |
| VGG16    | 81.5%    | 512.6MB| 2.2ms         |
| AlexNet  | 75.2%    | 217.9MB| 1.2ms         |
| LeNet-5  | 42.8%    | 0.2MB  | 0.8ms         |

### Performance Comparisons
![Model Performance Comparisons](https://github.com/jefftl/Capstone-Project-2024/blob/main/images/comp_graphs.png)
*Top left: Model accuracy comparison. Top right: Inference time comparison. Bottom left: Model size comparison. Bottom right: Accuracy vs inference time scatter plot.*

*Top left: Model accuracy comparison. Top right: Inference time comparison. Bottom left: Model size comparison. Bottom right: Accuracy vs inference time scatter plot.*

### Confusion Matrices
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/comp_confusion_matrix.png">
   </div>
*Confusion matrices for all four models showing classification performance across 30 plant species. Top left: LeNet-5, Top right: AlexNet, Bottom left: VGG16, Bottom right: ResNet18.*

### Performance Visualizations
- Model accuracy comparison charts
- Inference time comparisons
- Model size comparisons
- Confusion matrices for all models
- Training curves showing loss and accuracy progression

### Key Observations
- ResNet18 shows the best overall performance with highest accuracy
- VGG16 demonstrates good accuracy but requires significant storage
- AlexNet provides reasonable performance with moderate resource requirements
- LeNet-5's simple architecture proves inadequate for this complex task

## CONTACT DETAILS
- Jeffrey Lewis
- jeffreytudorlewis@gmail.com

