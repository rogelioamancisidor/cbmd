def load_dataset_modalities(num_accepts = 9807,num_acc_cls = 6537, num_rej=50000,scale='maxmin',dset='LC',x2_idx=[55], balanced = False, path='../../bimodal/data/'):
    import gzip, pickle
    from sklearn import preprocessing
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn.decomposition import PCA as sklearnPCA
    import numpy as np
    from sklearn.utils import shuffle
    import random

    # both dataset: lending club and santander purchase prediction       
    if dset == 'LC':    
        with gzip.open(path+'behavior_dta.pk.gz', 'rb') as f:
            behavior_dta = pickle.load(f, encoding='latin')

        X2  = behavior_dta['behaviour']

        with gzip.open(path+'application_dta.pk.gz', 'rb') as f:
            application_dta = pickle.load(f, encoding='latin')

        X1  = application_dta['data']

        idx_X2 = x2_idx
        X2  = X2[:,idx_X2]

        ''' put x1 and x2 together as X_a
            remember the 1 column is date '''
        date_a  = X1[:,0]
        X1      = X1[:,1:]

        X_a = np.c_[X1,X2]

        y_a = data_supervised_c['labels'][:,0]

        ''' select only one proportion of data from 2013 '''
        # no sepparate calibration samples
        # after October 2012
        X_2013 = X_a[date_a>=20121001,:]
        y_2013 = y_a[date_a>=20121001]

        my_seed = random.randint(1,100000)
        #my_seed = 12345
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6196, random_state=my_seed)
        for train_idx, rest in sss.split(X_2013, y_2013):
            X_tr_2013   = X_2013[train_idx]
            y_tr_2013   = y_2013[train_idx]

        X_a = np.r_[X_tr_2013,X_a[date_a<20121001,:]]
        y_a = np.r_[y_tr_2013,y_a[date_a<20121001]]
    
        # shuffle the concatenation
        X_a,y_a = shuffle(X_a,y_a)

    elif dset == 'SAN':
        with gzip.open(path+'santander_dta.pk.gz', 'rb') as f:
            data_supervised = pickle.load(f, encoding='latin')
        X_a = data_supervised['application']
        y_a = data_supervised['labels']


    # Create the Scaler object
    if scale == 'standard':
        scaler = preprocessing.StandardScaler()
    elif scale == 'maxmin':
        scaler = preprocessing.MinMaxScaler()
    elif scale == 'pca':
        scale = sklearnPCA(n_components=X_a.shape[1])
    elif scale == 'normmaxmin':    
        scaler = preprocessing.StandardScaler()
        X_a = scaler.fit_transform(X_a)
        scaler = preprocessing.MinMaxScaler()
    elif scale == 'maxminnorm':
        scaler = preprocessing.MinMaxScaler()
        X_a = scaler.fit_transform(X_a)
        scaler = preprocessing.StandardScaler()
    X_a = scaler.fit_transform(X_a)
    
    """ step 2 split 5% calibration, 70% CBMD, 25% test """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
    for train_idx, test in sss.split(X_a, y_a):
        X_tr, X_te  = X_a[train_idx], X_a[test]
        y_tr, y_te  = y_a[train_idx], y_a[test]

    if balanced:
        # create balanced set in the imputer data keeping all bads
        X_1 = X_tr[y_tr==1,:]
        y_1 = y_tr[y_tr==1]

        X_0 = X_tr[y_tr==0,:]
        y_0 = y_tr[y_tr==0]
        
        idx_good  = random.sample(range(0,X_0.shape[0]),int(num_accepts/2))
        idx_bads  = random.sample(range(0,X_1.shape[0]),int(num_accepts/2))

        X_tr = np.r_[X_1[idx_bads,:],X_0[idx_good,:]]
        y_tr = np.r_[y_1[idx_bads],y_0[idx_good]]

        X_tr,y_tr = shuffle(X_tr,y_tr)

    f.close()
    data = (X_tr,y_tr,X_te,y_te)

    return data
