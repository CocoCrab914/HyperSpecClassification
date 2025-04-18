### 🚀 Arguments
#### --image: Path to the hyperspectral image (GeoTIFF format) [/Data/02.tif] 

#### --samples: Path to a CSV file containing labeled training data [/Data/prunedsamples.csv]
#### Format: first column = label, remaining columns = features

#### --output: Path to save the classified output map

### ▶️ How to Run
#### Run the script:

```bash
python "path\Code\RF_Hyperspectral_Image_Classification.py" --image "path\Data\02.tif" --samples "path\Data\prunedsamples.csv" --output "path\Output\classified_output.tif"
```
#### classified image (both TIFF and PNG formats) will be saved in the Output/ directory.
