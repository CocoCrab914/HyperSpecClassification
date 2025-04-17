# 🌱 Random Forest Hyperspectral Image Classification
### A lightweight script for performing supervised classification on hyperspectral images using a Random Forest classifier.
### Designed for research or applied remote sensing tasks involving vegetation, soil, or land use mapping.

## 📦 Research Information
### This code is part of a software toolkit for ecological and environmental element extraction, specifically serving as the fine-scale vegetation classification module. It uses a Random Forest algorithm to classify hyperspectral imagery, ultimately generating thematic vegetation classification maps.

### The classification approach and algorithmic design can be found in the following publication: https://doi.org/10.1016/j.jag.2021.102398


## 🧠 Requirements
### Python 3.8 + 
#### Libraries: numpy, pandas, rasterio, scikit-learn, matplotlib

## 🚀 Arguments
### --image: Path to the hyperspectral image (GeoTIFF format)

### --samples: Path to a CSV file containing labeled training data
### Format: first column = label, remaining columns = features

### --output: Path to save the classified output map


## 📄 License
### This project is licensed under the MIT License.
