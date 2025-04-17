#!/bin/bash

# Download example hyperspectral image from Zenodo

URL="https://zenodo.org/records/15237072/files/02.tif?download=1"
DEST_DIR="data"
DEST_FILE="$DEST_DIR/example_input.tif"

# Create destination folder if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the file
echo "Downloading example input file from Zenodo..."
wget -O "$DEST_FILE" "$URL"

# Uncomment to use curl instead:
# curl -L "$URL" -o "$DEST_FILE"

echo "Download complete: $DEST_FILE"
