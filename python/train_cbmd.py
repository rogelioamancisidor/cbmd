# -*- coding: utf-8 -*-
from CBMD import CBMD 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json, pickle
import os
# supress all tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import wandb
from get_data import load_dataset_modalities

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim2", default=8,help="Dimensionality of modality 2", type=int)
    parser.add_argument("--ydim", default=2, help="Number of categories in the label y", type=int)
    parser.add_argument("--lambda_val", default=4000, help="Scaling parameter for balance the KL loss", type=int)
    parser.add_argument("--omega", default=0.8, help="Parameter for MI optimization ", type=float)
    parser.add_argument("--alpha", default=20.0,help="Parameter for classification loss", type=float)
    parser.add_argument("--layers_size_enc", default=[100],nargs='+', help="No of units in the hidden layer", type=int)
    parser.add_argument("--layers_size_enc2", default=[100],nargs='+', help="No of units in the hidden layer", type=int)
    parser.add_argument("--layers_size_dec", default=[200,200],nargs='+', help="No of units in the hidden layer", type=int)
    parser.add_argument("--layers_size_prior", default=[100],nargs='+',help="No of units in then hidden layer", type=int)
    parser.add_argument("--layers_size_cls", default=[100,100],nargs='+',help="No of units in then hidden layer", type=int),
    parser.add_argument("--dropout_enc", default=0,help="Dropout rate to be used in all hidden layers",type=float)
    parser.add_argument("--dropout_enc2", default=0,help="Dropout rate to be used in all hidden layers",type=float)
    parser.add_argument("--dropout_dec", default=0,help="Dropout rate to be used in all hidden layers",type=float)
    parser.add_argument("--dropout_prior",default=0,help="Dropout rate to be used in all hidden layers",type=float)
    parser.add_argument("--dropout_cls", default=0,help="Dropout rate to be used in all hidden layers", type=float)
    parser.add_argument("--latent_dim", default= 50,help="Dimensionality of the latent space", type=int)
    parser.add_argument("--n_samples", default= 5,help="No of samples that the encoder draws", type=int)
    parser.add_argument("--no_runs", default= 1, help="No of cross-validations to run", type=int)
    parser.add_argument("--epochs", default= 1000, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--save_every", default= 300, help="No of epochs to save the model", type=int)
    parser.add_argument('--scaling', default='maxmin', help='how to scale dataset')
    parser.add_argument("--batch_size", default= 100, help="No of samples that the encoder draws", type=int)
    parser.add_argument("--decoder_type", default="Gaussian",help="Name of the distribution assumed in the decoder", type=str)
    parser.add_argument("--dset", default="LC",help="Name of the data set", type=str)
    parser.add_argument('--x2_idx', default=[9, 11, 27, 50, 57, 62, 66, 70],nargs='+', type=int, help='index for x2. only for LC')
    parser.add_argument('--num_accepts', default=9807, type=int, help='number observations (equal number of goods and bads). Default values correspont to LC data.')
    parser.add_argument('--num_acc_cls', default=6537, type=int, help='number observations to train classifier (equal number of goods and bads). Default values correspont to LC data.')
    parser.add_argument("--outfile", default="lc_test",help="Name of the output folder", type=str)
    parser.add_argument('--wandb_mode', default='disabled', choices=['online','offline','disabled'], help='whether to log run in wandb')
    parser.add_argument('--wandb_user', default=None, help='your username at wandb, if you have.')
    parser.add_argument('--balance',action='store_false',help='default is True. downsampling the majority class')
    
    args = parser.parse_args()
    print (args)

    X_tr,y_tr, X_te,y_te = load_dataset_modalities(num_accepts=args.num_accepts, num_acc_cls=args.num_acc_cls, scale=args.scaling, dset=args.dset, x2_idx=args.x2_idx, balanced=args.balance)
        
    # select number of features in each modality
    # nr_x1 is fixed for LC, while nr_x2 is fixed for SAN
    if args.dset == 'LC':
        nr_x1 = 18  
        nr_x2 = len(args.x2_idx) + nr_x1
    elif args.dset == 'SAN':
        nr_x1 = 200 - args.dim2   
        nr_x2 = 200
    
    # choose x1 and x2
    x2_tr = X_tr[:,nr_x1:]
    x1_tr = X_tr[:,0:nr_x1]
    x1_te = X_te[:,0:nr_x1]
    x2_te = X_te[:,nr_x1:]
    
    wandb.init(project='cbmd', group=args.outfile, name=args.wandb_user, mode=args.wandb_mode)
    wandb.config.update(args)
    
    # Iterate over epochs.
    start = time.time()
    
    output_folder = "../output/"+str(args.outfile)
    try:
        os.mkdir(output_folder)
    except OSError:
        print ("Creation of the directory %s failed" % output_folder)
    else:
        print ("Successfully created the directory %s " % output_folder)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_metric = tf.keras.metrics.Mean()
    
    epochs = args.epochs
    auc_cbmd = np.zeros((epochs,args.no_runs))
    recon_x2 = np.zeros((epochs,args.no_runs))
    data = {}
    r = 0

    print('loading data...')
    # load data
    BUFFER_SIZE = x1_tr.shape[0]
    y_tr = tf.keras.utils.to_categorical(y_tr, num_classes=args.ydim).astype(np.float32)
    
    while r < args.no_runs:
        # save data to dic
        data[r] = {'x1_tr':x1_tr,
                   'x2_tr':x2_tr,
                   'y_tr':y_tr,
                   'x1_te':x1_te,
                   'x2_te':x2_te,
                   'y_te':y_te
                   }
        
        print('run cv %s out of %s' % (r+1, args.no_runs))
        print('building CBMD...')
        model = CBMD(
                    args.dim2,
                    args.ydim,
                    args.lambda_val,
                    args.omega,
                    args.alpha,
                    layers_size_enc=args.layers_size_enc,
                    layers_size_enc2=args.layers_size_enc2,
                    layers_size_dec=args.layers_size_dec,
                    layers_size_prior=args.layers_size_prior,
                    layers_size_cls=args.layers_size_cls,
                    dropout_enc=args.dropout_enc,
                    dropout_enc2=args.dropout_enc2,
                    dropout_dec=args.dropout_dec,
                    dropout_prior=args.dropout_prior,
                    dropout_cls=args.dropout_cls,
                    latent_dim=args.latent_dim,
                    n_samples=args.n_samples,
                    decoder_type=args.decoder_type,
                    name='cbmd',
                    )
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, output_folder+"/ckpts_"+str(r) , max_to_keep=3)

        # restore model if exits
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            save_path = manager.latest_checkpoint
            # increse checkpoint  
            checkpoint.step.assign_add(1)
            print("Model restored from {}. Trained for {} epochs".format(save_path,int(checkpoint.step)))
            assert (int(checkpoint.step)==args.epochs),'The model is already trained. Increase number of epochs or specify a different outfile parameter.'
        else:
            print("Model initializing from scratch.")
        
        tr_data = tf.data.Dataset.from_tensor_slices((x1_tr,x2_tr, y_tr)).shuffle(BUFFER_SIZE).batch(args.batch_size)
       
        print('num test labels', y_te.sum())
        while int(checkpoint.step) < args.epochs:
            for i, (x1_batch,x2_batch,y_batch) in enumerate(tr_data):
                costs, obj_fn = model.train((x1_batch,x2_batch,y_batch),optimizer)

                if np.isnan(obj_fn):
                    raise RuntimeError('cost is nan')

                loss_metric(obj_fn) # this saves the average loss during training

            pi_hat = model.classify(x1_te)['pi_hat']
            x2_hat = model.generate(x1_te)['x2_hat'].numpy()
            
            recon_x2[int(checkpoint.step),r] = np.sqrt(np.mean((x2_hat-x2_te)**2)) 
            
            error = roc_auc_score(y_te, pi_hat[:,1].numpy())
            auc_cbmd[int(checkpoint.step),r] = error
            
            wandb.log({'auc_cbmd': error, 'loss_cbmd':loss_metric.result()})
                
            if int(checkpoint.step) % 100 == 0:
                print ('epoch %s: error rate %0.4f' % (int(checkpoint.step),error)) 

            if int(checkpoint.step) % args.save_every == 0:
                save_path = manager.save(checkpoint_number=int(checkpoint.step)+1)
                print("Saved checkpoint for step {}: {}".format(int(checkpoint.step)+1, save_path))
        
            # increse checkpoint  
            checkpoint.step.assign_add(1)

        # save model from last epoch if doesnt exit
        if int(save_path.split('-')[1]) != int(checkpoint.step):
            save_path = manager.save(checkpoint_number=int(checkpoint.step))
            print("Saved latest checkpoint {}".format(save_path))

        pi_hat = model.classify(x1_te, training=False)['pi_hat'].numpy()
        error = roc_auc_score(y_te, pi_hat[:,1])
        print ('final error rate %0.4f' % (error)) 

        # increse r
        r+=1
        
    avg_auc = np.mean(auc_cbmd[-1,:])
    plt.figure()
    plt.plot(auc_cbmd)
    plt.title("auc {0:.4f}".format(avg_auc))
    plt.savefig(output_folder+'/auc_cbmd.pdf')
    
    avg_recon = np.mean(recon_x2[-1,:])
    plt.figure()
    plt.plot(recon_x2)
    plt.title("rmse {0:.4f}".format(avg_recon))
    plt.savefig(output_folder+'/recon_cbmd.pdf')
    
    with open(output_folder +'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(output_folder +'/data.pk', 'wb') as f:
        pickle.dump(data,f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elapsed_time = time.time() - start
    print ('time elapsed %f' % elapsed_time)
    
if __name__ == "__main__":
    train()
