### 🔽 Download Example Input Image

An example hyperspectral image (`.tif`, ~300 MB) is provided via [Zenodo](https://zenodo.org/records/15237072).

To download it automatically, 

Open [Zenodo](https://zenodo.org/records/15237072) or

run:

```bash
bash Download_image.sh
```

### 🔽 Extract pruned sample spectra

You can use the provided prunedsample.csv to run the classification code directly.

To extract your sample spectra, the customer must draw their sample polygons (.shp files) in ArcGIS or QGIS. Then, run Sample_spectra_extraction_from_polygons.py to retrieve the corresponding band information. 
