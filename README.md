# Single-cell-Classification-for-early-detection-of-Acute-Myeloid-Leukemia-Using-Deep-Learning-Models
Acute Myeloid Leukemia (AML) is one of the most life-threatening type of blood cancers, and its accurate classification is considered and remains a challenging task due to the visual similarity between various cell types.

This study addresses the classification of the multiclasses of AML cells Utilizing three deep learning models: ResNet50, YOLOv12, and Inception-ResNet50 v2. We applied two segmentation approaches based on cell and nucleus features, using the Hue channel and Otsu thresholding techniques to preprocess the images before classification.

Our experiments demonstrate that YOLOv12 with Otsu thresholding on cell-based segmentation achieved the highest level of validation and test accuracy, both reaching 99.3\%. ResNet50 with nucleus-based Otsu thresholding resulted in test accuracy of 93.87\% , while Inception-ResNet50 v2 with Hue channel cell-based segmentation reached 97.6\% test accuracy.

Combining effective segmentation with modern deep learning models demonstrated significant results and improvements in AML classification. This study emphasizes the importance of preprocessing in handling challenging cell types, particularly for cells, for instance, basophils and monocytes. 
