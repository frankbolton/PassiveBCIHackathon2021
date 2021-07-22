import neptune.new as neptune
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

import os
import mne
import pandas as pd
from mne.externals.pymatreader import read_mat
from sys import argv

# from sktime.classification.compose import ColumnEnsembleClassifier
# from sktime.classification.dictionary_based import BOSSEnsemble
# from sktime.classification.interval_based import TimeSeriesForestClassifier
# from sktime.classification.shapelet_based import MrSEQLClassifier
# from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from sklearn.utils import shuffle
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

data_path = 'C:\\Users\\frank\\code\\NeuroErgonomics_Hackathon_2021'
n_subs = 15
n_sessions = 2
diff = ['MATBeasy', 'MATBmed', 'MATBdiff']
ch_slice = ['F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'AF3', 'AFz', 'AF4','FP1', 'FP2', 'FPz']


def runModel(n_estimators):
    run = neptune.init(project='frankbolton/PassiveBCIHackathon2021', source_files=[__file__, argv[0], 'environment.yaml'])
    # run = neptune.init(project='frankbolton/helloworld', source_files=[__file__, argv[0], 'environment.yaml'])

    data_params =   {'n_estimators': n_estimators,
                    }

    params =        {'verbose':1,
                    }

    run['data_params'] = data_params
    run['params'] = params
    run["sys/tags"].add(['sktime', 'loop3', 'tsfresh', 'randomforest'])

    accuracies = list()
    # for sub_n, session_n in itertools.product(range(n_subs), range(n_sessions)):
    for sub_n in range(n_subs):
        run['subject'].log(sub_n)
        session_n = 0
        epochs_data = []
        labels = []
        for lab_idx, level in enumerate(diff):
            sub = 'P{0:02d}'.format(sub_n+1)
            sess = f'S{session_n+1}'
            path = os.path.join(os.path.join(data_path, sub), sess) + f'/eeg/alldata_sbj{str(sub_n+1).zfill(2)}_sess{session_n+1}_{level}.set'
            # Read the epoched data with MNE
            epochs = mne.io.read_epochs_eeglab(path, verbose=False)
            # You could add some pre-processing here with MNE
            # We will just select some channels (mostly frontal ones)
            epochs = epochs.drop_channels(list(set(epochs.ch_names) -set(ch_slice)))

            # Get the data and concatenante with others MATB levels
            tmp = epochs.get_data()
            epochs_data.extend(tmp)
            labels.extend([lab_idx]*len(tmp))
        
            X = np.array(epochs_data)
    #         print(X.shape)
        labels = np.array(labels)
        y = labels
        x_names = [f'dim_{x}' for x in range(X.shape[1])]    
        X_df = pd.DataFrame(columns = x_names)

        sample =  0 
        for sample in range(X.shape[0]):
            data = X[sample,:,:]
            list_of_series = []
            for xx in range(X.shape[1]):
                list_of_series.append(X[sample,xx,:])

            X_df = X_df.append(pd.DataFrame([list_of_series], columns = x_names))

    #     # print(len(list_of_series))
        X_df.reset_index(drop = True)
    #     #     display(X_df)


        session_n = 1
        epochs_data = []
        labels = []
        for lab_idx, level in enumerate(diff):
            sub = 'P{0:02d}'.format(sub_n+1)
            sess = f'S{session_n+1}'
            path = os.path.join(os.path.join(data_path, sub), sess) + f'/eeg/alldata_sbj{str(sub_n+1).zfill(2)}_sess{session_n+1}_{level}.set'
            # Read the epoched data with MNE
            epochs = mne.io.read_epochs_eeglab(path, verbose=False)
            # You could add some pre-processing here with MNE
            # We will just select some channels (mostly frontal ones)
            epochs = epochs.drop_channels(list(set(epochs.ch_names) -set(ch_slice)))

            # Get the data and concatenante with others MATB levels
            tmp = epochs.get_data()
            epochs_data.extend(tmp)
            labels.extend([lab_idx]*len(tmp))
        
            X_s2 = np.array(epochs_data)
    #         print(X_s2.shape)
        labels = np.array(labels)
        y_s2 = labels


        Xs2_df = pd.DataFrame(columns = x_names)

        sample =  0 
        for sample in range(X_s2.shape[0]):
            data = X_s2[sample,:,:]
            list_of_series = []
            for xx in range(X_s2.shape[1]):
                list_of_series.append(X_s2[sample,xx,:])

            Xs2_df = Xs2_df.append(pd.DataFrame([list_of_series], columns = x_names))

    #     # print(len(list_of_series))
        Xs2_df.reset_index(drop = True)
    #     display(Xs2_df)
            
        X_train, X_test, y_train, y_test = X_df, Xs2_df, y, y_s2
        
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        
    #     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        # steps = [
        # ("concatenate", ColumnConcatenator()),
        # ("classify", TimeSeriesForestClassifier(n_estimators=data_params['n_estimators'], random_state=0, n_jobs=-1)),
        # ]
        # clf = Pipeline(steps)

        clf = make_pipeline(
        ColumnConcatenator(), TSFreshFeatureExtractor(n_jobs=-1, show_warnings=False), RandomForestClassifier(n_estimators=data_params['n_estimators'], random_state=0, n_jobs=-1)
        )

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)
        run['test/acc'].log(acc)

    run.stop()
    return(np.array(accuracies).mean())
