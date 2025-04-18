import os
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping, Polygon
from tqdm import tqdm

def extract_spectra(shapefile: gpd.GeoDataFrame, hyperspectral_stack: str, num_polygons: int):
    """Extract per-pixel hyperspectral spectra from raster stack for each field polygon."""
    geoms = shapefile.geometry.values

    # Apply a 10m inward buffer to reduce edge effects
    for i in range(num_polygons):
        geoms[i] = Polygon(geoms[i].buffer(-10))

    spectra = np.zeros((1, 1))  # will append later
    pixel_counts = np.zeros(num_polygons)

    for idx in range(num_polygons):
        geom = [mapping(geoms[idx])]
        with rio.open(hyperspectral_stack) as src:
            if geoms[idx].area <= 0:
                continue
            out_image, _ = mask(src, geom, crop=True)

        # Handle no-data and invalid values
        out_image = out_image.astype("float32")
        out_image[out_image >= 255] = np.nan
        out_image[out_image <= 0] = np.nan

        # Flatten spatial dimensions, keep spectral bands
        bands, rows, cols = out_image.shape
        flattened = out_image.reshape(bands, -1)  # shape: (bands, pixels)

        pixel_counts[idx] = flattened.shape[1]
        spectra = np.append(spectra, flattened, axis=1)

    spectra = np.delete(spectra, 0, axis=1)  # remove dummy column
    return pd.DataFrame(spectra.T), pixel_counts

def main():
    hyperspectral_stack = "F:/data_Iran/Iran_S2_L2A_201801-201911/stack/ndvi_stack_newmask111.tif"

    crop_info = [
        ("caohaitong", 15),
        ("gouyagencao", 13),
        ("danyemanjing", 13),
        ("hongtu", 13),
        ("stone", 6),
        ("other", 4)
    ]

    for crop_name, num_polygons in crop_info:
        print(f"Processing: {crop_name}")
        shapefile_path = f"E:/classification/sample/{crop_name}.shp"
        shapefile = gpd.read_file(shapefile_path)

        spectra_df, pixel_counts = extract_spectra(shapefile, hyperspectral_stack, num_polygons)

        output_path = f"E:/classification/sample/output/{crop_name}.csv"
        spectra_df.to_csv(output_path, index=False)
        print(f"Saved {spectra_df.shape[0]} spectra × {spectra_df.shape[1]} bands to {output_path}")

if __name__ == "__main__":
    main()
