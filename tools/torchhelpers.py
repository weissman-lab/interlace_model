# These are helper functions to train NN models using pytorch
import torch
import os
import re
import shutil
import gc

# ----------------------------------------------------------------
# -------------------------------- Model Class
# ----------------------------------------------------------------

class MultiTaskNNModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2, layers_data, final_activation1 = 'sigmoid', final_activation2 = 'sigmoid'):
        '''
        The layers data is a list of tuples for each layer (type, size, activation, dropout)
        Just linear layers for now.
        '''
        super().__init__()
        activation_dict = {'relu' : torch.nn.ReLU(), 
                            'sigmoid' : torch.nn.Sigmoid()}
        self.last_activation1 = activation_dict[final_activation1]
        self.last_activation2 = activation_dict[final_activation2]
        l = []
        l_size_last = None
        for l_num, (l_type, l_size, l_activation, l_dropout) in enumerate(layers_data):
            l_input = input_dim if l_num == 0 else layers_data[l_num - 1][1]
            l_output = l_size
            l_size_last = l_output
            l.append(torch.nn.Linear(l_input, l_output))
            if l_num != len(layers_data) - 1:
                l.append(torch.nn.BatchNorm1d(l_output))
            if l_activation != 'identity' and l_num != len(layers_data) - 1:
                l.append(activation_dict[l_activation])
            if l_dropout > 0:
                l.append(torch.nn.Dropout(l_dropout))
        self.nnmodel = torch.nn.Sequential(*l)
        # Now add on final multi-part output layer
        self.final1 = torch.nn.Linear(l_size_last, output_dim1)
        self.final2 = torch.nn.Linear(l_size_last, output_dim2)
    def forward(self, x, task_id):
        res = self.nnmodel(x)
        if task_id == 1:
            res = self.last_activation1(self.final1(res))
        elif task_id == 2:
            res = self.last_activation2(self.final2(res))
        else:
            assert False, 'Bad Task ID passed'
        return(res)

# ----------------------------------------------------------------
# -------------------------------- Training Function
# ----------------------------------------------------------------

def train_multitask_with_early_stopping(model, target_name, train_loader1, val_loader1, train_loader2, val_loader2, device, writer, tmp_dir, init_layers = True, learn_rate = 1e-3, patience = 10, min_delta = 0.001, max_epochs = 50, save_ckpts = False):
    loss_function1 =  torch.nn.BCELoss() # removed weights for these #torch.nn.MSELoss() #torch.nn.BCEWithLogitsLoss() #torch.nn.BCELoss()##torch.nn.CrossEntropyLoss()  # ##
    loss_function2 =  torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate, weight_decay = 1e-6, amsgrad = True)
    best_val_loss = float('inf')
    best_val_model = None
    patience_counter = 0
    # Set initial weights for model
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) and init_layers == True:
            layer.weight = torch.nn.init.xavier_normal_(layer.weight)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        epoch_fname_this = f'{tmp_dir}epoch_{epoch}.pth'
        epoch_fname_last = f'{tmp_dir}epoch_{epoch-1}.pth'
        if os.path.isfile(epoch_fname_this) and save_ckpts == True:
            print(f'Loading epoch {epoch} checkpoint from file just for validation, skipping training.', flush = True)
            checkpoint = torch.load(epoch_fname_this, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #epoch = checkpoint['epoch']
            loss1 = checkpoint['loss1']
            loss2 = checkpoint['loss2']
        else:
            if (save_ckpts == True and epoch != 0 and os.path.isfile(epoch_fname_last)):
                print(f'Loading last saved epoch {epoch-1} checkpoint from file to resume training.', flush = True)
                checkpoint = torch.load(epoch_fname_last, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #epoch = checkpoint['epoch']
                loss1 = checkpoint['loss1']
                loss2 = checkpoint['loss2']
            print(f'Now training on epoch {epoch}.', flush = True)
            model.train()
            for (X1, y1), (X2, y2) in zip(train_loader1, train_loader2):
                X1, y1 = X1.to(device), y1.to(device)
                X2, y2 = X2.to(device), y2.to(device)
                # Forward pass  
                output1 = model(X1, task_id = 1)
                output2 = model(X2, task_id = 2)
                # Calculating the loss function
                loss1 = loss_function1(output1, y1)
                loss2 = loss_function2(output2, y2)
                loss = loss1 + loss2
                # Backward pass  
                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (valX1, valy1), (valX2, valy2) in zip(val_loader1, val_loader2):
                valX1, valy1 = valX1.to(device), valy1.to(device)
                valX2, valy2 = valX2.to(device), valy2.to(device)
                val_outputs1 = model(valX1, task_id = 1)
                val_outputs2 = model(valX2, task_id = 2)
                val_loss1 = loss_function1(val_outputs1, valy1).item()
                val_loss2 = loss_function2(val_outputs2, valy2).item()
                val_loss += val_loss1 + val_loss2
        val_loss /= len(val_loader1)
        print(f"Epoch {epoch:>3d}, Total Training Loss: {loss.item():>5f}, Training Loss Task 1/2: {loss1.item():>5f}/{loss2.item():>5f}, Total Validation Loss: {val_loss:>5f}, Validation Loss Task 1/2: {val_loss1:>5f}/{val_loss2:>5f}", flush = True)
        writer.add_scalars(f"Loss:{target_name}", {'train_loss' : loss.item() ,
                                                  'val_loss' : val_loss} , epoch)
        writer.flush()
        # save checkpoint to file if doesn't exist
        if save_ckpts == True and not os.path.isfile(epoch_fname_this):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss1': loss1,
                'loss2': loss2
            }, epoch_fname_this)
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_val_model = model
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping! No improvement for {patience} epochs.")
            model = best_val_model
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            break
    # If we actually finish training through all epochs, clean up at the end if hasn't been deleted already
    if os.path.exists(tmp_dir) and epoch == (max_epochs -1):
        shutil.rmtree(tmp_dir)

    
# ----------------------------------------------------------------
# -------------------------------- Other Helper Functions
# ----------------------------------------------------------------
def generate_tuning_grid(layer_counts, layer_sizes, layer_dropouts):
    '''NB. These choices refer to the *hidden* layers, while the input and output 
    layers are determined by the shape of the data.
    Create list of layer_lists to create different model architectures
    currently making all of them in a "funnel" shape with tuple:
    (type, size, activation, dropout)'''
    tuning_grid_list = []
    for count in layer_counts:
        if count < 1:
            print('Error: Must have at least 1 hidden layer', flush = True)
            return None
        for size in layer_sizes:
            for dropout in layer_dropouts:
                types = ['linear'] * count
                final_size = round(size/count)
                step_size = round((size - final_size) / count) + 1
                sizes = list(range(final_size, size, step_size))
                sizes = sizes + [size]
                sizes.sort(reverse = True)
                sizes = sizes[:count]
                #print(f"{count},{len(sizes)}")
                activations = ['relu'] * (count)
                #print(f"{count}+{activations}")
                dropouts = [dropout] * (count - 1) + [0.0] # don't use dropout in final layer
                nn_arch = list(zip(types, sizes, activations, dropouts))
                tuning_grid_list.append(nn_arch)
    return(tuning_grid_list)

def logits_to_probs(x):
    '''Convert a tensor of logits to probabilities.'''
    s = torch.nn.Sigmoid()
    return s(x)
