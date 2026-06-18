#!/usr/bin/env bash
set -e  # stop on error

echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

echo "Step 1: Generate synthetic data"
quarto render 01_generate_synthetic_data.qmd

echo "Step 2: Extract symptoms"
python3 02_extract_symptoms.py

echo "Step 3: Vectorize notes"
python3 03_vectorize_notes.py

echo "Step 4: Compile dataset"
quarto render 04_compile.qmd

echo "Step 5: Predict"
python3 05_predict.py

echo "✅ Pipeline complete"