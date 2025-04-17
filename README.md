# HyperSpecClassification
🌱 Random Forest Hyperspectral Image Classification
A lightweight script for performing supervised classification on hyperspectral images using a Random Forest classifier.
Designed for research or applied remote sensing tasks involving vegetation, soil, or land use mapping.

📦 Features
Supports hyperspectral .tif input images

Band selection (pruning to selected spectral features)

Training/validation with external sample CSV

Random Forest classification with scikit-learn

Outputs classified GeoTIFF map

Simple imputation for missing values

Easting/Northing pixel-to-coordinate mapping

🧠 Requirements
Python 3.8+

Libraries: numpy, pandas, rasterio, scikit-learn, matplotlib

You can install dependencies via:

bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install numpy pandas rasterio scikit-learn matplotlib
🚀 Usage
bash
Copy
Edit
python classify_rf.py \
  --image path/to/image.tif \
  --samples path/to/training_samples.csv \
  --output path/to/output_map.tif
Arguments
--image: Path to the hyperspectral image (GeoTIFF format)

--samples: Path to a CSV file containing labeled training data
Format: first column = label, remaining columns = features

--output: Path to save the classified output map

📁 Example Training CSV Format

label	b3	b4	b5	...
caohaitong	...	...	...	...
danyemanjing	...	...	...	...
🗺️ Output
A classified .tif file using integer codes:

1 = caohaitong

2 = danyemanjing

3 = gouyagencao

4 = hongtu

5 = other

6 = none

📄 License
This project is licensed under the MIT License.
See the LICENSE file for full details.

🤝 Contributing
Feel free to fork, improve, and submit a pull request!

