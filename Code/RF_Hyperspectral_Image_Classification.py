#!/usr/bin/env python
# Author: dkk
# Description: Image classification using Random Forest

# Import necessary libraries: file I/O, data manipulation, machine learning, and image processing
import os
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import imageio

# Define color palette (RGB) for visualizing each class label
label_colors = {
    1: [16, 93, 39],     # Class 1: Greenish
    2: [86, 179, 69],    # Class 2: Light Green
    3: [176, 240, 126],  # Class 3: Pale Green
    4: [188, 59, 37],    # Class 4: Red
    5: [246, 181, 60],   # Class 5: Yellow
    6: [0, 0, 0]         # Class 6: Black (background or unclassified)
}

# Function to read a multi-band raster image and extract its array, geotransform, and metadata
def read_image(path):
    with rio.open(path) as img:
        img_pre = img.read(masked=True)  # Read with masking for nodata
        img_arr = img_pre.filled()       # Convert masked values to array with fill values
        img_meta = img.profile
        img_gt = img_meta["transform"]
        img_crs = img_meta["crs"]
    return img_arr, img_gt, img_crs, img_meta

# Select a subset of meaningful bands from the original image, and handle NaNs
def extract_pruned_bands(img_arr):
    band_indices = list(range(3, 9)) + list(range(28, 49))  # Choose selected bands
    pruned = np.zeros((len(band_indices), img_arr.shape[1], img_arr.shape[2]))
    for i, idx in enumerate(band_indices):
        pruned[i, :, :] = img_arr[idx, :, :]
    pruned[np.isnan(pruned)] = 0
    return pruned, [f"b{idx}" for idx in band_indices]

# Convert 3D image array to a flat pandas DataFrame, and add spatial coordinates
def make_data_frame(img_arr, col_names, gt):
    shp = img_arr.shape
    data = img_arr.flatten().reshape(shp[0], shp[1] * shp[2]).T
    df = pd.DataFrame(data, columns=col_names)
    df = df.where(df != 0)  # Keep 0 as missing
    df['Easting'] = pixel_id_to_lon(gt, np.arange(df.shape[0]) % shp[2])
    df['Northing'] = pixel_id_to_lat(gt, np.repeat(np.arange(shp[1]), shp[2]))
    return df

# Convert pixel row indices to geographic coordinates using affine transform
def pixel_id_to_lat(gt, y): return gt[4] * y + gt[5]
def pixel_id_to_lon(gt, x): return gt[0] * x + gt[2]

# Train a Random Forest model using given hyperparameters
def random_forest_classifier(train_x, train_y, options=None):
    clf = RandomForestClassifier(**options) if options else RandomForestClassifier()
    clf.fit(train_x, train_y)
    return clf

# Map class label strings to integer values for classification
def label_to_int(preds, label_map):
    return np.vectorize(label_map.get)(preds)

# Save the predicted classification as a GeoTIFF using original metadata
def save_classification_map(preds, shape, meta, output_file):
    preds = preds.reshape(shape[1:])  # Reshape flat predictions into 2D
    meta.update(dtype=rio.uint8, count=1)  # Update metadata for single band
    with rio.open(output_file, 'w', **meta) as dst:
        dst.write(preds.astype(rio.uint8), 1)

# Save the classification result as a color-coded PNG for visualization
def save_colored_png(preds, shape, output_png, label_colors):
    preds = preds.reshape(shape[1:])  # shape: (height, width)
    rgb_image = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
    
    for label, color in label_colors.items():  # Assign color to each label
        rgb_image[preds == label] = color

    imageio.imwrite(output_png, rgb_image)
    print(f"Colored PNG saved to {output_png}")

# Main pipeline: load data, train model, predict, and export results
def main(args):
    # Step 1: Load the hyperspectral image
    print("Reading image...")
    img_arr, img_gt, _, img_meta = read_image(args.image)

    # Step 2: Extract selected bands to reduce dimensionality
    print("Extracting features...")
    pruned_img_arr, band_names = extract_pruned_bands(img_arr)

    # Step 3: Create a DataFrame from image and add geographic information
    print("Preparing dataframe...")
    df = make_data_frame(pruned_img_arr, band_names, img_gt)

    # Step 4: Load the labeled sample data from CSV
    print("Reading samples...")
    ds = pd.read_csv(args.samples)
    target = ds.iloc[:, 0].values
    train = ds.iloc[:, 1:].values.astype('float32')
    print("Dataset size: ", train.shape)

    # Step 5: Split labeled data into training and testing sets
    print("Training/Validation split...")
    train_x, test_x, train_y, test_y = train_test_split(
        train, target, test_size=0.2, random_state=0)

    # Step 6: Impute missing values using a constant fill strategy
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='constant')
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.transform(test_x)
    print("Train Dataset size: ", train_x.shape)
    print("Test Dataset size: ", test_x.shape)
    
    # Step 7: Train a Random Forest model using specified options
    print("Training Random Forest model...")
    options = {
        "n_estimators": 200,
        "max_features": 'sqrt',
        "oob_score": True,
        "n_jobs": -1,
        "random_state": 42
    }
    model = random_forest_classifier(train_x, train_y, options)

    # Step 8: Evaluate model performance and print metrics
    print("Evaluating model...")
    preds = model.predict(test_x)
    print(f"Train Accuracy: {accuracy_score(train_y, model.predict(train_x))}")
    print(f"Test Accuracy: {accuracy_score(test_y, preds)}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_y, preds))

    # Step 9: Predict labels for the entire image
    print("Predicting full image...")
    df_imputed = imputer.transform(df[band_names])
    full_preds = model.predict(df_imputed)

    # Step 10: Map string labels to integer codes
    label_map = {
        'caohaitong': 1,
        'danyemanjing': 2,
        'gouyagencao': 3,
        'hongtu': 4,
        'other': 5,
        'none': 6
    }
    preds_int = label_to_int(full_preds, label_map)

    # Step 11: Save the prediction as a GeoTIFF file
    print("Saving classification map...")
    save_classification_map(preds_int, pruned_img_arr.shape, img_meta, args.output)
    print(f"Classification map saved to {args.output}")
    
    # Step 12: Also export a color-encoded PNG for easy visualization
    png_output = args.output.replace('.tif', '.png')
    save_colored_png(preds_int, pruned_img_arr.shape, png_output, label_colors)

# Define command-line arguments and trigger the classification pipeline
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest Hyperspectral Image Classification')
    parser.add_argument('--image', required=True, help='Path to hyperspectral image (.tif)')
    parser.add_argument('--samples', required=True, help='Path to training sample CSV file')
    parser.add_argument('--output', required=True, help='Output path for classification map (.tif)')
    args = parser.parse_args()
    main(args)
