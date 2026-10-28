# INTERLACE - dIagNosTic Excellence for oldeR aduLts in primAry CarE

This repository accompanies the manuscript *Artificial intelligence-based diagnostic decision support for primary care of older adults using collective intelligence and imitation learning*. The full citation is here:

Streiffer CD, Press MJ, Bishop NS, Schmid BE, Sliwa A, Vyas AL, Ungar LH, Peifer M, Weissman GE. Artificial intelligence-based diagnostic decision support for primary care of older adults using collective intelligence and imitation learning. _J Am Med Inform Assoc_ 2026. https://doi.org/10.1093/jamia/ocag132

or in BibTex format:

```bibtex
@article{weissman2026multitask,
  author = {Streiffer, Christopher D. and Press, Matthew J. and Bishop, Nicholas S. and Schmid, Benjamin E. and Sliwa, Alyssa and Vyas, Anuradha L. and Ungar, Lyle H. and Peifer, MaryAnne and Weissman, Gary E.
  title = {Artificial intelligence-based diagnostic decision support for primary care of older adults using collective intelligence and imitation learning},
  journal = {J Am Med Inform Assoc},
  year = {2026},
  month = {September},
  page = {ocag132},
  url = {https://doi.org/10.1093/jamia/ocag132}
}

```

# Repository Description

This repository contains resources to replicate a data pipeline that predicts diagnoses and orders for older adults in the primary care setting. The `00_run_all.sh` file will run each file sequentially. It will not work until you change the paths for Metamap and BioClinicalBERT, and it is recommended to run each file individually the first time after you change the data source to your data. Each file contains notes on what each section of code does.

Step 1: Generate Synthetic Data (`01_generate_synthetic_data.qmd`) - this file creates synthetic patient data that mirrors the SQL Clarity extract and structured data processing. If adapting this to your dataset, you will need to replace this file with your data extract and reformatting. Please note that this file adjusts the synthetic data to result in final predictions that are representative of the training data.

Step 2: Extract Symptoms Information (`02_extract_symptoms.py`) - this file extracts symptoms from notes using Metamap. Please note that you will need to update the path to the Metamap server. You can find more information about it here: http://metamap.nlm.nih.gov

Step 3: Unstructured Data Processing (`03_vectorize_notes.py`) - this file vectorizes unstructured data using BioClinicalBERT. It is recommended to remove PHI from your text data prior to using this tool. Please note that you will need to update the paths for BioClinicalBERT to the location of the model. You can find more information about it here: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

Step 4: Compile (`04_compile.qmd`) - this file merges the data files together, imputes missing data, and standardizes all data. The resulting output will be ready to be run through the model.

Step 5: Predict (`05_predict.py`) - this file reads in the neural network model and predicts on the dataset completed in step 4.
