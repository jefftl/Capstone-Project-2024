# Datasheet

I obtained the [Plant Classification](https://www.kaggle.com/datasets/marquis03/plants-classification) dataset from kaggle. I unfortunately have very little information on the dataset. So in some sections I will have to make asumptions however I will make it clear when these assumptions are being made. 

## Motivation

- <b>For what purpose was the dataset created?</b> There is no mention of why the dataset was created, however I suspect it was for a simalair CNN project. 
- <b>Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset?</b> The dataset was created by a kaggle user under the tag [Marquis03](https://www.kaggle.com/marquis03). There is no indication that it was on behalf of an entity or externally funded. The creator has contributed a variety of image classification datasets on his kaggle page. 

 
## Composition

- **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** The dataset is comprised of 30,000 .png images split into train, test and validate folders. Each image represents one of 30 different types of plants, including fruits, vegetables, and other agricultural plants. There are also 3 csv files that contain the title of each of the images in the train, test and validate folders.
- **How many instances of each type are there?**
  * 21,000 training images
  * 3,000 validation images
  * 6,000 test images
  * These are distributed over 30 different types of plants.
- <b>Is there any missing data?</b> There are images within the dataset that are incorrectly labeled, however I have no way of determining how many without going through all the pictures manually or using another pre-trained dataset and reviewing images the dataset labels incorrectly. 
- <b>Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by    doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?</b> No data in this dataset would be considered confidential. 

## Collection process

- <b>How was the data acquired?</b> There is no indication of how the dataset was obtained, however I assume it was obtained using web scraping techniques.  
- <b>If the data is a sample of a larger subset, what was the sampling strategy?</b> The data was already sampled, however I have provided: [split-dataset.ps1](https://github.com/jefftl/Capstone-Project-2024/blob/main/src/split-dataset.ps1) to split the images into train, test and validate folders. The splitting happens randomly with each folder containing a percentage of images stipulated by the user. 
- <b>Over what time frame was the data collected?</b> Ther is no indication on the time taken to collect the dataset. 

## Preprocessing/cleaning/labelling

- <b>Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.</b> The dataset was already organised into train, validate and test folders when obtained. The data underwent different transfoms when being loaded into the CNN model. These include:
  * Resizing
  * Random croping
  * Random horizontal flip
  * Random rotation
  * Color jitter
  * For more detailed information please look at each of the model cards.
- <b>Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?</b> The raw data was saved as the transforms only changed the data when loaded into the model. 
 
## Uses

- <b>What other tasks could the dataset be used for?</b>
  * Plant species identification apps
  * Development of farming and gardening applications
  * Development of plant disease identification applications
- <b>Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?</b>  Without detailed information about the image collection process, lighting conditions, or geographical distribution of the plants, users should be aware that the model's performance might vary in different real-world conditions.
- <b>Are there tasks for which the dataset should not be used? If so, please provide a description.</b>
  * Critical agricultural or botanical decisions without additional verification
  * Commercial plant identification without proper testing and validation

## Distribution

- <b>How has the dataset already been distributed?</b> The dataset is publicly available on kaggle. 
- <b>Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?</b> The dataset is distributed under the Apache 2.0 license, which allows for both personal and commercial use with proper attribution. 

## Maintenance

- <b>Who maintains the dataset?</b> The dataset is not being maintained and it is likely that it will never be updated. 

