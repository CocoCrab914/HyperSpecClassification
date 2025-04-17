#!/usr/bin/env python
# Author: dkk
# Description: Image classification using Random Forest

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


def read_image(path):
    with rio.open(path) as img:
        img_pre = img.read(masked=True)
        img_arr = img_pre.filled()
        img_meta = img.profile
        img_gt = img_meta["transform"]
        img_crs = img_meta["crs"]
    return img_arr, img_gt, img_crs, img_meta


def extract_pruned_bands(img_arr):
    band_indices = list(range(3, 8)) + list(range(28, 49))
    pruned = np.zeros((len(band_indices), img_arr.shape[1], img_arr.shape[2]))
    for i, idx in enumerate(band_indices):
        pruned[i, :, :] = img_arr[idx, :, :]
    pruned[np.isnan(pruned)] = 0
    return pruned, [f"b{idx}" for idx in band_indices]


def make_data_frame(img_arr, col_names, gt):
    shp = img_arr.shape
    data = img_arr.flatten().reshape(shp[0], shp[1] * shp[2]).T
    df = pd.DataFrame(data, columns=col_names)
    df = df.where(df != 0)
    df['Easting'] = pixel_id_to_lon(gt, np.arange(df.shape[0]) % shp[2])
    df['Northing'] = pixel_id_to_lat(gt, np.repeat(np.arange(shp[1]), shp[2]))
    return df


def pixel_id_to_lat(gt, y): return gt[4] * y + gt[5]
def pixel_id_to_lon(gt, x): return gt[0] * x + gt[2]


def random_forest_classifier(train_x, train_y, options=None):
    clf = RandomForestClassifier(**options) if options else RandomForestClassifier()
    clf.fit(train_x, train_y)
    return clf


def label_to_int(preds, label_map):
    return np.vectorize(label_map.get)(preds)


def save_classification_map(preds, shape, meta, output_file):
    preds = preds.reshape(shape[1:])
    meta.update(dtype=rio.uint8, count=1)
    with rio.open(output_file, 'w', **meta) as dst:
        dst.write(preds.astype(rio.uint8), 1)


def main(args):
    print("Reading image...")
    img_arr, img_gt, _, img_meta = read_image(args.image)
    print("Extracting features...")
    pruned_img_arr, band_names = extract_pruned_bands(img_arr)

    print("Preparing dataframe...")
    df = make_data_frame(pruned_img_arr, band_names, img_gt)

    print("Reading samples...")
    ds = pd.read_csv(args.samples)
    target = ds.iloc[:, 0].values
    train = ds.iloc[:, 1:].values.astype('float32')

    print("Training/Validation split...")
    train_x, test_x, train_y, test_y = train_test_split(
        train, target, test_size=0.2, random_state=0)

    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='constant')
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.transform(test_x)

    print("Training Random Forest model...")
    options = {
        "n_estimators": 200,
        "max_features": 'sqrt',
        "oob_score": True,
        "n_jobs": -1,
        "random_state": 42
    }
    model = random_forest_classifier(train_x, train_y, options)

    print("Evaluating model...")
    preds = model.predict(test_x)
    print(f"Train Accuracy: {accuracy_score(train_y, model.predict(train_x))}")
    print(f"Test Accuracy: {accuracy_score(test_y, preds)}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_y, preds))

    print("Predicting full image...")
    df_imputed = imputer.transform(df[band_names])
    full_preds = model.predict(df_imputed)

    label_map = {
        'caohaitong': 1,
        'danyemanjing': 2,
        'gouyagencao': 3,
        'hongtu': 4,
        'other': 5,
        'none': 6
    }
    preds_int = label_to_int(full_preds, label_map)
    print("Saving classification map...")
    save_classification_map(preds_int, pruned_img_arr.shape, img_meta, args.output)
    print(f"Classification map saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest Hyperspectral Image Classification')
    parser.add_argument('--image', required=True, help='Path to hyperspectral image (.tif)')
    parser.add_argument('--samples', required=True, help='Path to training sample CSV file')
    parser.add_argument('--output', required=True, help='Output path for classification map (.tif)')
    args = parser.parse_args()
    main(args)
