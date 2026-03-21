import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
warnings.filterwarnings('ignore')
import time

from model import Model,CustomDenoiser
from dataset import load_dataset, get_eval,mean_std
from diffusion_utils import  impute_mask,refinement
import json



parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=-1, help='GPU index.')
parser.add_argument('--split_idx', type=int, default=0, help='Split idx.')
parser.add_argument('--ratio', type=str, default=30, help='Masking ratio.')
parser.add_argument('--hid_dim', type=int, default=32, help='Hidden dimension.')
parser.add_argument('--mask', type=str, default='MCAR', help='Masking machenisms.')
parser.add_argument('--num_trials', type=int, default=10, help='Number of sampling times.')
parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')
parser.add_argument('--batch_size', type=int, default=8192, help='Batch size for training.')
parser.add_argument('--epochs', type=int,default=10001, help='Epochs to train')
parser.add_argument('--method_name', type=str,default="RefiDiff", help='Method name')

args = parser.parse_args()

# Check CUDA and set device
if torch.cuda.is_available():
    if args.gpu == -1:  # Use all available GPUs
        device = torch.device('cuda')
        gpu_ids = list(range(torch.cuda.device_count()))
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    else:  # Use specific GPU
        device = torch.device(f'cuda:{args.gpu}')
        gpu_ids = [args.gpu]
        print(f"Using GPU: {args.gpu}")
else:
    device = torch.device('cpu')
    gpu_ids = []
    print("Using CPU")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    dataname = args.dataname
    split_idx = args.split_idx
    hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = args.ratio
    num_trials = args.num_trials
    num_steps = args.num_steps
    print("Dataset: ",dataname," split_idx= ",split_idx, " Mask type= ",mask_type)
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask, cat_bin_num = load_dataset(dataname, split_idx, mask_type, ratio)
    

    mean_X, std_X = mean_std(train_X, train_mask)
    in_dim = train_X.shape[1]


    X = (train_X - mean_X) / std_X / 2
    X = torch.tensor(X)

    X_test = (test_X - mean_X) / std_X / 2
    X_test = torch.tensor(X_test)

    mask_train = torch.tensor(train_mask)
    mask_test = torch.tensor(test_mask)

    ckpt_dir = f'ckpt/{args.method_name}/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
    os.makedirs(ckpt_dir, exist_ok=True)
    X_miss = (1. - mask_train.float()) * X
    X_miss=refinement(rec_X=X_miss.cpu().numpy(),
                      mask=(1. - mask_train.float().cpu().numpy()),
                      len_num=train_num.shape[1])                                           
    train_data = X_miss


    batch_size = args.batch_size
    train_data = torch.tensor(X_miss, dtype=torch.float32)
    train_mask_tensor = torch.tensor(mask_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_mask_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = args.epochs

    denoise_fn = CustomDenoiser(in_dim, hid_dim).to(device)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters denoise func: ", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)
    
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model wrapped with DataParallel across GPUs: {gpu_ids}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=40, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):

        batch_loss = 0.0
        len_input = 0
        for batch in train_loader:
            inputs, batch_mask = batch
            inputs=inputs.float().to(device)
            batch_mask=batch_mask.float().to(device)
            loss = model(inputs, batch_mask)

            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, f'{ckpt_dir}/model.pt')
            
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {curr_loss:.4f}, Best loss: {best_loss:.4f}')

                break
        if epoch%500==0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {curr_loss:.4f}, Best loss: {best_loss:.4f}')

        if epoch % 1000 == 0:
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, f'{ckpt_dir}/model_{epoch}.pt')

    

    rec_Xs = []

    for trial in range(num_trials):
    
        X_miss = (1. - mask_train.float()) * X
        X_miss = X_miss.to(device)
        impute_X = X_miss

        in_dim = X.shape[1]

        denoise_fn = CustomDenoiser(in_dim, hid_dim).to(device)

        model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)
        
        checkpoint = torch.load(f'{ckpt_dir}/model.pt', map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        net = model.denoise_fn_D
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        
        

        num_samples, dim = X.shape[0], X.shape[1]
        rec_X = impute_mask(net, impute_X, mask_train, num_samples, dim, num_steps, device)
        
        mask_int = mask_train.to(torch.float).to(device)
        rec_X = rec_X * mask_int + impute_X * (1-mask_int)
        rec_Xs.append(rec_X)
        

    rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 
    len_num = train_num.shape[1]

    rec_X=refinement(rec_X=rec_X.cpu().numpy(),
                     mask=(1. - mask_train.float().cpu().numpy()),
                     len_num=train_num.shape[1]
                     )   
    rec_X = rec_X* 2
    X_true = X.cpu().numpy() * 2

    pred_X = rec_X[:]
    len_num = train_num.shape[1]

    res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
    pred_X[:, len_num:] = res

    mae, rmse, acc = get_eval(dataname, pred_X, X_true, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)


    print('in-sample, MAE: ',mae," RMSE: ", rmse, "ACC: ", acc)



    rec_Xs = []

    X_miss = (1. - mask_test.float()) * X_test
    X_miss=refinement(rec_X=X_miss.cpu().numpy(),
                      mask=(1. - mask_test.float().cpu().numpy()),
                      len_num=train_num.shape[1]
                    )
       
    X_miss=torch.from_numpy(X_miss)
    X_miss = X_miss.to(device)
    impute_X = X_miss
    for trial in range(num_trials):
        in_dim = X_test.shape[1]
        denoise_fn = CustomDenoiser(in_dim, hid_dim).to(device)
        model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)
        checkpoint = torch.load(f'{ckpt_dir}/model.pt', map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        net = model.denoise_fn_D
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        
        num_samples, dim = X_test.shape[0], X_test.shape[1]
        rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
        
        mask_int = mask_test.to(torch.float).to(device)
        rec_X = rec_X * mask_int + impute_X * (1-mask_int)
        rec_Xs.append(rec_X)
        

    rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 
    rec_X=refinement(rec_X=rec_X.cpu().numpy(),
                     mask=(1. - mask_test.float().cpu().numpy()),
                     len_num=train_num.shape[1]
                     ) 
    rec_X = rec_X * 2
    X_true = X_test.cpu().numpy() * 2

    pred_X = rec_X[:]
    len_num = train_num.shape[1]
    res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
    pred_X[:, len_num:] = res

    mae_out, rmse_out, acc_out = get_eval(dataname, pred_X, X_true, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask,oos=True)
    
    print('out-of-sample MAE: ', mae_out, "RMSE: ", rmse_out, "ACC: ", acc_out)
