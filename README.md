# Single-cell Classification for early detection of Acute Myeloid Leukemia Using YOLOv12, Inception-ResNet-v2, and ResNet50 Deep Learning Models

## Note: The dataset is not included in this repository. you can find it in this Kaggle link: https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection/data%7D%7Bthis
Acute Myeloid Leukemia (AML) is one of the most life-threatening type of blood cancers, and its accurate classification is considered and remains a challenging task due to the visual similarity between various cell types.

This study addresses the classification of the multiclasses of AML cells Utilizing three deep learning models: ResNet50, YOLOv12, and Inception-ResNet50 v2. We applied two segmentation approaches based on cell and nucleus features, using the Hue channel and Otsu thresholding techniques to preprocess the images before classification.

Our experiments demonstrate that YOLOv12 with Otsu thresholding on cell-based segmentation achieved the highest level of validation and test accuracy, both reaching 99.3\%. ResNet50 with nucleus-based Otsu thresholding resulted in test accuracy of 93.87\% , while Inception-ResNet50 v2 with Hue channel cell-based segmentation reached 97.6\% test accuracy.

Combining effective segmentation with modern deep learning models demonstrated significant results and improvements in AML classification. This study emphasizes the importance of preprocessing in handling challenging cell types, particularly for cells, for instance, basophils and monocytes. 

<img width="2929" height="1512" alt="detailed_model_aml_final" src="https://github.com/user-attachments/assets/659644f1-a7d1-4b19-98f1-1300d439660f" />
