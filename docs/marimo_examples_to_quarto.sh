#!/bin/bash

# Define the directory to scan
TARGET_DIR="notebooks/examples"
text_to_remove="import marimo as mo"

for file in "$TARGET_DIR"/*; do
  if [ -f "$file" ]; then

    base_name="$(basename "$file" .py)"

    echo "Processing $file -> ${base_name}.ipynb and ${base_name}.qmd"

    marimo export ipynb "$file" -o "docs/assets/marimo/$base_name.ipynb" && \
    quarto convert "docs/assets/marimo/$base_name.ipynb" --output "docs/03_Examples/$base_name.qmd"
    sed -i "/$text_to_remove/d" "docs/03_Examples/$base_name.qmd"

    if [ $? -eq 0 ]; then
      echo "Successfully processed $file."
    else
      echo "Error processing $file."
    fi
  fi
done

rm docs/assets/marimo/*.ipynb 

echo "All files processed!"
