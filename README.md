# The Future of Fashion Retail: Deploying Computer Vision for Enhanced Swapaholic Operations

## Introduction
<img width="752" alt="image" src="https://github.com/user-attachments/assets/4bdb4054-dfbc-484a-a545-1871d9595b53">

*<div align="center">Figure 1: Current Swapaholic UI</div>*

With the ongoing AI wave, the current merchant sphere is evolving rapidly. Hence, businesses  need to stay competitive and respond to the rapidly changing market conditions. Swapaholic, one of the biggest platforms in Singapore for buying and selling clothes between consumers, has root problems with cumbersome data consolidation processes and an inefficient online marketplace. The main problem of the platform is that it does not use unified data from multiple sources, limiting its ability to leverage information efficiently. As a result, the company is unable to respond in time to consumer demands and get insights from their behaviour, which in turn leads to a lower competitiveness and attractiveness to customers.

An article from “Scaling Up Excellence” highlights the pervasive data fragmentation as a prominent cause for issues that retail businesses face. It states that the data should be unified so the business can update information about a product without any delays and inaccuracies. (Morsen, 2023).

Further compounding these issues is the labour-intensive and inefficient seller onboarding and listing process. Due to cumbersome factors such as time-consuming manual inputs for listing, the user experience for sellers is likely to be unpleasant. It takes around 10 minutes to create one card for an item which increases the average session duration on the website. However, the funnel abandonment rate also increases in tandem, leading to lower consumption of the website in the future. This issue degrades user experience and decreases the operational effectiveness of the marketplace. Hence, the implementation of CV technologies has the potential to overcome all the problems and lower the number of labour units that need to operate the system.

The project aims to discover the best performing ML model among the ones highlighted above, to integrate into Swapaholic’s user interface using CV techniques and automate the apparel listing process in its website. This will help to reduce, and eventually avoid the manual effort required from the seller to manually enter details of his or her listing, and the accuracy of the details of each listing is expected to improve as well. Overall, we are not only aiming to boost operational efficiency within Swapaholic, but we are also looking to increase their user retention and user satisfaction metrics, generating growth in Swapaholic’s overall user sales.

One such example of a successful implementation of process automation in the retail sector using CV can be observed in Amazon Go. The company was a pioneer in this field, where they utilised CV and deep learning algorithms to decrease the amount of theft, track customers, and obtain data of their customers’ shopping behaviour habits, while decreasing management costs as well (Honaman, 2024). A CV implementation in monotonous and complex human tasks is key to handling many problems at once.
Taking inspiration from Amazon Go, the project hopes that integration of CV techniques into Swapaholic’s operations will contribute to a faster and more efficient way of product categorization.

## Dataset
In our research, we used the dataset “iMaterialist (Fashion) 2020 at FGVC7” from the Kaggle website. This dataset consists of 48826 files that contain files related to clothing and accessories. Segmentation and classification tasks are performed using the labels each image has, and these labels are stored in the value ImageId. We use the version of 2020 as the dataset was expanded and attributes were correctly relabeled since the 2019 version.

The dataset consists of images of people wearing clothes in different poses, it also contains the masks and annotations for them. The masks are used to make accurate segmentation between images.

The dataset is 23.7 GB in size and includes:

1. Images folder
2. JSON file containing category and attribute label descriptions
3. csv file containing annotations

The csv file has 4 columns:

1. “ImageId” is a label for an image
2. “EncodedPixels” are run-length encoded masks with pixel-wise annotations 
3. “ClassId” is an ID for each mask which represents the category
4. “AttributesIds” are the IDs 

## Mask2Former Model Architecture
### Description
Mask2Former is a unified framework for segmentation tasks, including instance, semantic, and panoptic segmentation. It leverages a transformer-based architecture to predict segmentation masks directly, using attention mechanisms to model pixel-to-object relationships.

### Data Pre-processing
The data preprocessing pipeline implements a FashionpediaDataset class, featuring two key components: an ID mapping system and a dataset processor. The IDMapper class handles the conversion of arbitrary category and attribute IDs to consecutive integers, maintaining bidirectional mappings that can be saved and loaded for consistency across sessions. The FashionpediaDataset class inherits from PyTorch's Dataset class and processes image data alongside their corresponding segmentation masks, category labels, and attribute annotations. Images are resized to 384x384 pixels and normalised using ImageNet statistics, while segmentation masks are converted from run-length encoding to binary masks and interpolated to match the image dimensions. The pipeline generates one-hot encoded attribute labels for each segmentation and the dataset is split into 70% train, 15% validation and 15% test. A custom collation function is implemented for batch processing, outputting a dictionary containing tensor batches of pixel values, segmentation masks, category labels, and attribute labels, ready for model training.

### Loss Functions
The loss functions implementation presents a multi-task learning framework that employs uncertainty-based task weighting for simultaneous multi-class category classification, multi-label attribute prediction, and instance segmentation. The architecture utilises learnable uncertainty parameters through the UncertaintyWeights module, which dynamically balances the relative importance of each task by learning log variances that are converted to precision weights. The loss computation incorporates Hungarian matching to optimise:

1. Jaccard Loss (1 - mIoU) for mask predictions
<img width="682" alt="image" src="https://github.com/user-attachments/assets/d326e9c4-1c6a-4fa4-953b-c18176761c7f">

2. Cross-entropy loss for category predictions
3. Binary cross-entropy loss for attribute set predictions

The Hungarian matching algorithm is employed to optimise the assignment between predictions and ground truth values, minimising the total loss:

<img width="659" alt="image" src="https://github.com/user-attachments/assets/413c5997-9a08-4656-842e-97f0154607ad">

The final loss function combines these task-specific losses using the learned uncertainty weights, with an added regularisation term to prevent the model from trivially minimising losses by increasing uncertainties. This approach allows the model to automatically adapt the contribution of each task to the total loss based on their relative difficulties and uncertainties, potentially leading to more balanced and effective multi-task learning.

### CustomMask2FormerForFashionpedia model
We implemented a custom extension of Mask2Former for the Fashionpedia dataset by modifying the architecture to simultaneously handle instance segmentation, category classification, and attribute set prediction. The model builds upon the pre-trained Mask2Former-Swin-Large backbone (frozen during training) from the Hugging Face Hub and adds two task-specific linear classifier heads for category and attribute set prediction, operating on the transformer decoder's last hidden states. To balance the multi-task learning objectives, we incorporated an uncertainty weighting mechanism that dynamically adjusts the relative contributions of segmentation, category classification, and attribute set prediction losses. The architecture maintains Mask2Former's original segmentation capabilities while extending its functionality to handle multi-class category classification and multi-label attribute classification through parallel classification heads.

### Training

![image](https://github.com/user-attachments/assets/27ffd6ea-1242-4cdd-b1b2-6da8cc84ce52)
*<div align="center">Figure 2: Training vs Validation Loss for Mask2Former</div>*

The training pipeline implements the CustomMask2FormerForFashionpedia model that leverages the Hungarian algorithm for optimal bipartite matching between predicted and ground truth masks, ensuring efficient assignment of segment predictions to target labels. Using an AdamW optimizer with a conservative learning rate of 1e-5 for stable convergence, the pipeline incorporates checkpointing and memory optimization features. It employs gradient accumulation for efficient batch processing and includes comprehensive memory management through systematic garbage collection and CUDA cache clearing. The pipeline loads model weights from previous checkpoints when available and processes training data in batches, computing losses for both categorical and attribute predictions across pixel-level masks, with the Hungarian matching algorithm minimising pairwise losses between predictions and ground truth. During each epoch, the model alternates between training and validation phases, tracking respective losses while maintaining memory efficiency. The system automatically saves model states, optimizer configurations, and loss metrics at designated checkpoints, enabling training continuity and performance monitoring. Additionally, the pipeline includes visualisation capabilities for tracking training and validation loss curves over time, facilitating model performance analysis across training epochs.

### Evaluation
The evaluation pipeline implements a comprehensive assessment framework for a multi-task segmentation model that simultaneously handles mask prediction, category classification, and attribute detection. The pipeline loads the best-performing model checkpoint based on validation loss and evaluates it on test data in batches to manage memory efficiently. For each batch, it computes three key metrics:

1. Mean Intersection over Union (mIoU) for mask predictions
2. Dice coefficient for category predictions
3. Dice coefficient for attribute set predictions

The attribute and category Dice calculations account for variable numbers of predictions and labels by computing pairwise scores and selecting the best matches using the Hungarian algorithm. To prevent memory overflow during evaluation of large datasets, the pipeline implements strategic memory management through garbage collection and CUDA memory clearing. Results are processed in configurable batch sizes and saved to a text file, with the evaluation defaulting to a subset of 1024 samples for efficiency.

### Visualisation

<img width="1458" alt="image" src="https://github.com/user-attachments/assets/995c56f1-75de-449a-b5a4-70800741eb40">

*<div align="center">Figure 3: Sample Segmentation for Mask2Former</div>*

The visualisation pipeline implements an instance segmentation workflow that processes images through the CustomMask2FormerForFashionpedia model, incorporating both category classification and attribute set prediction. The system first generates mask predictions with confidence scores, which are then refined through non-maximum suppression (NMS) to eliminate overlapping detections using an IoU threshold of 0.5. The pipeline processes the top-k predictions (initially 100, filtered to 15 after NMS) and interpolates the mask logits to match the original image dimensions. For each detected instance, the pipeline predicts both category labels and multiple attributes using softmax and sigmoid activations respectively. The final visualisation component overlays the predicted masks on the original image using distinct colours, accompanied by a legend displaying the predicted categories and up to five attributes per instance.

## A/B Testing
For our user testing, we employed Maze.co and two Figma mockups to assess the usability and user preference of listings UI: https://app.maze.co/report/Listing-AB/b2nx547m2gm3q19/intro 

1. The original manual Listing A 
2. The newly proposed automated Listing B

Participants were tasked to successfully navigate the uploading of a clothing image for listing twice; once using Listing A and once using Listing B. Out of the initial 52 participants, 25 (48%) fully completed all tasks. Outliers—such as those who spent over 685.84 seconds on the second listing or skipped tasks—were excluded from the analysis to focus on genuine user interaction data.
The findings showed a slight preference for the automated Listing B over the original Listing A.

<img width="1099" alt="image" src="https://github.com/user-attachments/assets/f37738d5-d59f-4ed5-86f9-c8e6b8ce38ac">

*<div align="center">Figure 4: Listing Method Survery Results</div>*

Using a 5-point scale of preference (where Listing A = 1 and Listing B = 5), we calculated a mean preference of 3.81, with both mode and median at 4. This shows a generally favourable response to Listing B. The automated Listing B also demonstrated efficiency gains, with a lower average completion time of 64 seconds compared to Listing A’s 109 seconds. Listing B also had a narrower spread in task completion times (standard deviation of 19.96 seconds versus Listing A's 33.54 seconds), suggesting a more consistent user experience.

Taken together, these results highlight several potential benefits of automated features in Listing B. The automation appears to streamline the process, reducing both completion time and variability, making it more convenient and user-friendly. The automated features in Listing B likely reduce cognitive load by minimising manual steps and decision points, allowing users to complete listings faster and with less effort. The lower standard deviation for Listing B’s completion times can also be attributed to the automated steps helping users encounter fewer unexpected interactions, resulting in smoother, more consistent experiences across participants.

##
