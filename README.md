# Generating Customer's Credit Behavior with Deep Generative Models
Minimal `tensorflow` re-implementation for the framework in **Generating Customer's Credit Behavior with Deep Generative Models:** [paper](https://www.sciencedirect.com/science/article/pii/S0950705122002532). If you are interested in the original code (implemented in Theano), contact me.

If you use this code in your research, please cite:

        @article{mancisidor2022generating,
                title={Generating Customer's Credit Behavior with Deep Generative Models},
                author={Mancisidor, Rogelio A and Kampffmeyer, Michael and Aas, Kjersti and Jenssen, Robert},
                journal={Knowledge-Based Systems},
                pages={108568},
                doi = {10.1016/j.knosys.2022.108568},
                year={2022},
                publisher={Elsevier}
        }

## Requirements
The code for CBMD is developed in `tensorflow==2.7.0`. It is possible to log model training with `weights & biases`, you just need to add your user name when training the model. In addition to `tensorflow==2.7.0`  and `wandb==0.12.7`, you need to dowload the following libraries: `numpy==1.21.3`, `sklearn==1.0.1`, and `matplotlib==3.5.0`.

The structure of the project should look like this:

```
cbmd
   │───data
   │───output
   │───python
```

Otherwise you will get error messages when loading the data, saving figures etc.

## Downloads
### Lending Club Dataset
The application data can be obtanied [here](https://biedu-my.sharepoint.com/:u:/g/personal/rogelio_a_mancisidor_bi_no/EcgPz45I3RVEu0NP6ZKFjwcBevv_UyPupOrOk2nGi7VGzQ?e=pvYQqg) and the behavior data [here](https://biedu-my.sharepoint.com/:u:/g/personal/rogelio_a_mancisidor_bi_no/ET0VGWiJlZ5Jgpj94f7JfkEBfSpVfJcB2p7aBZIsZgaunA?e=k7FdYo).

### Santander Dataset
The original data can be obtanied from [here](https://www.kaggle.com/c/santander-customer-transaction-prediction/data). In this other [version](https://biedu-my.sharepoint.com/:u:/g/personal/rogelio_a_mancisidor_bi_no/ETYsdKH2OFhPukow_jaMKW8Bineu6OS67s_ee7n__spRag?e=aJH83x), the features are sorted according with their predictive power in ascending order, i.e. first column is the least predictive feature, while last column is the most predictive feature as explained in the paper.  

## Usage
### Training

Make sure the [requirements](#requirements) are satisfied in your environment, and relevant [datasets](#downloads) are downloaded. `cd` into `python`, and run

```bash
python -u train_cbmd.py 
```

to train the CBMD model with the LC dataset.

See the file `running_specs.sh` for details about input parameters for the SAN dataset.

You can play with the hyperparameters using arguments, e.g.:
- **`--omega`**: Parameter for MI optimization
- **`--lambda_val`**: Scaling parameter for the KL loss 
- **`--zdim`**: dimension of the latent variable

For all arguments see all `add_argument()` functions in `train_cbmd.py`
