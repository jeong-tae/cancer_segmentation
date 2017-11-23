# Cancer_Segmentation
This repository aims to segment a cancer from CT images. The collection may not be used for commercial purposes. This collection is freely available to browse, download, and use for scientific and educational purposes as outlined in the [Creative Commons Attribution 3.0 Unported License.](https://creativecommons.org/licenses/by/3.0/)  See TCIA's [Data Usage Policies and Restrictions](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions) for additional details.

## Description
I tried Segmentation using RandomForest Approach. Which is little different from normal pixel-wise approach.

I use sliding window to extract spatial, local information from image. I hope this setting will helps to predict where cancer is. Finally, this model predict for each pixel but based on region of images.
