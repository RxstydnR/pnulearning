import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" 
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils.data import data_shuffle, get_data
from models.model import build_model
from PULoss import PULoss, PNULoss


def main():
    assert (opt.PNU and opt.nnPU) == False, "You can only use either PNU Learning or nnPU Learning."

    # Data preparation
    if opt.PNU:
        print("============= PNU Learning =============")

        # Positive label data (label: 1)
        P_X = get_data(opt.P_dataset)
        P_Y = np.ones(len(P_X))

        # Negative label data (label: -1)
        N_X = get_data(opt.N_dataset)
        N_Y = np.ones(len(N_X)) * (-1)

        # Unlabel data (label: 0)
        U_X = get_data(opt.U_dataset)
        U_Y = np.zeros(len(U_X))

        # Data shuffle
        P_X, P_Y = data_shuffle(P_X, P_Y)
        N_X, N_Y = data_shuffle(N_X, N_Y)
        U_X, U_Y = data_shuffle(U_X, U_Y)

        # Make dataset
        assert opt.P_n <= len(P_X), "Cannot set more P_n than the number of Positive data."
        assert opt.N_n <= len(N_X), "Cannot set more N_n than the number of Negative data."
        assert opt.U_n <= len(U_X), "Cannot set more U_n than the number of Unlabeled data."

        X_train = np.vstack((P_X[:opt.P_n], N_X[:opt.N_n], U_X[:opt.U_n]))
        Y_train = np.hstack((P_Y[:opt.P_n], N_Y[:opt.N_n], U_Y[:opt.U_n]))
    
    else:
        print("============= PU Learning =============")

        # Positive label data (label: 1)
        P_X = get_data(opt.P_dataset)
        P_Y = np.ones(len(P_X))

        # Unlabel data (label: -1)
        U_X = get_data(opt.U_dataset)
        U_Y = np.ones(len(U_X)) * (-1)

        # Data shuffle
        P_X, P_Y = data_shuffle(P_X, P_Y)
        U_X, U_Y = data_shuffle(U_X, U_Y)

        # Make dataset
        assert opt.P_n <= len(P_X), "Cannot set more P_n than the number of Positive data."
        X_train = np.vstack((P_X[:opt.P_n], U_X[:opt.U_n]))
        Y_train = np.hstack((P_Y[:opt.P_n], U_Y[:opt.U_n]))

    X_train,Y_train = data_shuffle(X_train,Y_train) 
    X_train = X_train.astype('float32')/255.
    print(f"Train: {X_train.shape,Y_train.shape}")

    # Model Preparation    
    model = build_model(model_name=opt.model, input_shape=X_train[0].shape)
    optimizer = Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if opt.PNU:
        print("PNU Loss is selected.")
        model.compile(
            loss=PNULoss(prior=opt.prior, eta=opt.eta, loss=opt.loss, temp=opt.temp),
            optimizer=optimizer)
    else:
        if opt.nnPU:
            print("nnPU Loss is selected.")
        else:
            print("PU Loss is selected.")
        
        model.compile(
            loss=PULoss(prior=opt.prior, nnPU=opt.nnPU, gamma=1, beta=0),
            optimizer=optimizer)

    # Training
    history = model.fit(
                X_train,Y_train, 
                epochs=opt.epochs, 
                batch_size=opt.batchsize,
                verbose=2)
    
    model.save(os.path.join(opt.save_dir,'model.h5'), include_optimizer=True)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(opt.save_dir, 'history.csv'))

    plt.figure(dpi=120)
    hist_df[['loss']].plot()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(opt.save_dir, 'Loss.png'))
    plt.close()
    plt.clf()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            prog='main.py',
            usage='PU or PNU Learning with deep learning model.',
            description='PU or PNU Learning experiment parameters.'
    )
    parser.add_argument('--save_dir', type=str, default="Results", help='path of the save results folder.')
    parser.add_argument('--P_dataset', required=True, help='Positive Labeled data path.')
    parser.add_argument('--N_dataset', type=str, default=None, help='Negative Labeled data path.')
    parser.add_argument('--U_dataset', required=True, help='Positive & Negative Unlabelled data path.')
    parser.add_argument('--P_n', type=int, default=0, help='the number of Positive labell data for training.')
    parser.add_argument('--N_n', type=int, default=0, help='the number of Negative labell data for training.')
    parser.add_argument('--U_n', type=int, default=2000, help='the number of Unlabelled data for training.')
    parser.add_argument('--PNU', action='store_true', default=False, help='PNU Learning: True')
    parser.add_argument('--nnPU', action='store_true', default=False, help='nnPU: True, uPU: False')
    parser.add_argument('--prior', type=float, default=0.4, help='prior distribution of positive data.')
    parser.add_argument('--loss', type=str, default='sigmoid', choices=['sigmoid','softmax'], help='eta in PNU Learning.')
    parser.add_argument('--eta', type=float, default=0.0, help='eta in PNU Learning.')
    parser.add_argument('--temp', type=float, default=1.0, help='temperature in sigmoid and softmax loss.')
    parser.add_argument('--model', type=str, default="ResNet18", choices=['ResNet18','DRN26','DenseNet121','CNN_paper','CNN'], help='deep learning model.')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--epochs', type=int, default=15, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    opt = parser.parse_args()

    assert opt.PNU != opt.nnPU, "Please select one of them."
    
    os.makedirs(opt.save_dir, exist_ok=False)
    main()