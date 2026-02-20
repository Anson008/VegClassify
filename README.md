# VegClassify

A geospatial analysis tool that automatically determines the optimal NDVI threshold for vegetation classification in 
NAIP imagery using a deep learning–based reference model.

The system leverages a trained deep learning model to generate pixel-wise vegetation classifications from RGB imagery. 
These predictions are treated as pseudo–ground truth and are used to compute the NDVI threshold that best aligns 
traditional index-based classification with model-based semantic segmentation.

## Overview

Vegetation classification is commonly performed using the Normalized Difference Vegetation Index (NDVI). 
However, selecting an appropriate NDVI threshold is often heuristic and scene-dependent.

This tool automates threshold selection by:

1. Running a deep learning model on NAIP RGB imagery to produce vegetation masks. 
2. Computing NDVI for the same imagery. 
3. Sweeping candidate NDVI thresholds. 
4. Comparing NDVI-based classifications with the deep learning output. 
5. Selecting the threshold that maximizes agreement between the two methods.

The result is a data-driven NDVI threshold optimized for the specific scene.

## Features
1. Automatic NDVI threshold selection 
2. GPU-accelerated deep learning inference 
3. Scene-specific optimization 
4. Pixel-level evaluation

## Installation
```commandline
git clone https://github.com/Anson008/VegClassify.git
cd VegClassify
conda env create -f vegclassify_environment.yaml
```
## Usage

## Example Output

## How to cite

Huaqing Wang, Xingchen Zhao, Simin Gholami, Christopher McGinty, Brent Chamberlain, Xiaojun Qi,
A hybrid deep learning and NDVI threshold approach for high-resolution urban greenspace classification,
Urban Forestry & Urban Greening,
Volume 118,
2026,
129332,
ISSN 1618-8667,
https://doi.org/10.1016/j.ufug.2026.129332.

## References

### Connected Components
1. [OpenCV Connected Component Labeling and Analysis](https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)
2. [Python OpenCV – Connected Component Labeling and Analysis](https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/)

### NDVI
1. [Calculate NDVI Using NAIP Remote Sensing Data in the Python Programming Language](https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/vegetation-indices-in-python/calculate-NDVI-python/)

### Image Thresholding
1. [OpenCV Image Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
2. [OpenCV Thresholding (cv2.threshold)](https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/)

### Image Segmentation
1. [Expectation-Maximization (EM) Algorithm: Solving a Chicken and Egg Problem](https://towardsdatascience.com/solving-a-chicken-and-egg-problem-expectation-maximization-em-c717547c3be2)
2. [Gaussian Mixture Model Selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
3. [Semantic Segmentation using mmsegmentation](https://mducducd33.medium.com/sematic-segmentation-using-mmsegmentation-bcf58fb22e42)

### Selenium
1. [Locating Elements](https://selenium-python.readthedocs.io/locating-elements.html#locating-by-xpath)
2. [How to use regular expressions in xpath in Selenium with python?](https://www.tutorialspoint.com/how-to-use-regular-expressions-in-xpath-in-selenium-with-python)
3. [Page Object Model and Page Factory in Selenium Python](https://www.browserstack.com/guide/page-object-model-in-selenium-python)
4. [Selenium with Python](https://selenium-python.readthedocs.io/index.html)

### Geospatial Raster Data
1. [Rioxarray Documentation](https://corteva.github.io/rioxarray/stable/index.html)
2. [Add lat and lon to DataArray read in by rioxarray](https://gis.stackexchange.com/questions/443801/add-lat-and-lon-to-dataarray-read-in-by-rioxarray)
3. [How to find coordinates of pixels of a GeoTIFF image with Python](https://gis.stackexchange.com/questions/394455/how-to-find-coordinates-of-pixels-of-a-geotiff-image-with-python)
4. [Raster Coordinate Reference Systems (CRS)](https://pygis.io/docs/d_raster_crs_intro.html)

### Morphological Operations
1. [Understanding Morphological Image Processing and Its Operations](https://towardsdatascience.com/understanding-morphological-image-processing-and-its-operations-7bcf1ed11756)
2. [Python OpenCV – Morphological Operations](https://www.geeksforgeeks.org/python-opencv-morphological-operations/#)

### Design Patterns
1. [Design Patterns - Filter Pattern](https://www.tutorialspoint.com/design_pattern/filter_pattern.htm)

### Miscellaneous
1. [What is the correct way to change image channel ordering between channels first and channels last?](https://stackoverflow.com/questions/43829711/what-is-the-correct-way-to-change-image-channel-ordering-between-channels-first)
2. [How to take website screenshots in Python](https://screenshotone.com/blog/how-to-take-website-screenshots-in-python/)
3. [ScreenshotOne Documentation](https://screenshotone.com/docs/getting-started/)
4. [EarthPy Documentation](https://earthpy.readthedocs.io/en/latest/index.html)
5. [What are the ranges to recognize different colors in RGB space?](https://stackoverflow.com/questions/42882498/what-are-the-ranges-to-recognize-different-colors-in-rgb-space)
6. [Getting Display resolution with python isn't accurate](https://stackoverflow.com/questions/73268410/getting-display-resolution-with-python-isnt-accurate)
7. [Compute Confusion Matrix](https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/compute-confusion-matrix.htm)
8. [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
