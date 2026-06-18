#Predict with the Neural Network

#Packages
import os
import pandas as pd
import torch
import pyarrow
from collections import OrderedDict
from pathlib import Path
from tools.torchhelpers import MultiTaskNNModel, generate_tuning_grid

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Establish Paths
DATA_DIR = Path('data/')
PREDICTIONS_DIR = Path('predictions/')
MODEL_OBJECT_DIR = 'resources/model_multitask-L3S1024D0.0r8.pth'

#Load Data
X_df = pd.read_csv(f"{DATA_DIR}/final_X_dt.csv")

#Convert to Float32 Tensors
X_df = torch.tensor(X_df.values, dtype=torch.float32)

#Model Parameters
input_dim_size = 2738
output_dim_size1 = 669 #diagnoses
output_dim_size2 = 1000 #orders
grid_cell = generate_tuning_grid([3], [1024], [0.0])[0]

print("Setup complete")

#Instantiate Model
model = MultiTaskNNModel(input_dim_size, output_dim_size1, output_dim_size2, grid_cell)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)

#Load Weights
state_dict = torch.load(MODEL_OBJECT_DIR, map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("module."):
        name = k[len("module."):]
    else:
        name = k
    new_state_dict[name] = v.float()
model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

print("Model loaded")

#Predict
with torch.no_grad():
    out1 = model(X_df.to(device), task_id=1)
    out2 = model(X_df.to(device), task_id=2)
    preds1 = out1.cpu().numpy()
    preds2 = out2.cpu().numpy()

preds_df1 = pd.DataFrame(preds1, columns=[f'prob_class_{i}' for i in range(preds1.shape[1])])
preds_df2 = pd.DataFrame(preds2, columns=[f'prob_class_{i}' for i in range(preds2.shape[1])])

#Save Predictions
preds_df1.to_parquet(os.path.join(PREDICTIONS_DIR, 'raw_preds_dxs.parquet'), index=False)
preds_df2.to_parquet(os.path.join(PREDICTIONS_DIR, 'raw_preds_orders.parquet'), index=False)
