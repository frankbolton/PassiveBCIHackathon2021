import numpy as np
import datetime
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

import os
import mne
import pandas as pd
from mne.externals.pymatreader import read_mat
from sklearn.utils import shuffle

import pickle
from sklearn.model_selection import train_test_split



data_path = '/home/frank/Documents/PassiveBCIHackathon2021'#'C:\\Users\\frank\\code\\NeuroErgonomics_Hackathon_2021'

# data_path = '/home/dcas/l.darmet/data/contest/comeptition_done'
n_subs = 15
diff = ['MATBeasy', 'MATBmed', 'MATBdiff']

ch_slice = ['F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'AF3', 'AFz', 'AF4','FP1', 'FP2', 'FPz']
# ch_slice = ['FP1', 'FP2', 'FPz']


n_estimators = 300 
narrow_feature_space = False


data_params =   {'n_estimators': n_estimators,
                    'narrow_feature_space': narrow_feature_space,
                }
    


all_results = pd.DataFrame({'epochID':np.arange(447)})

file_time = 'outputs/'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
for sub_n in range(n_subs):
    # Train
    print(f"{sub_n} - session 0")
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
        if (narrow_feature_space):
            epochs = epochs.drop_channels(list(set(epochs.ch_names) - set(ch_slice)))

        # Get the data and concatenante with others MATB levels
        tmp = epochs.get_data()
        epochs_data.extend(tmp)
        labels.extend([lab_idx]*len(tmp))
    print(len(labels))
    print(np.array(epochs_data).shape)


    print(f"{sub_n} - session 1")
    session_n = 1
    for lab_idx, level in enumerate(diff):
        sub = 'P{0:02d}'.format(sub_n+1)
        sess = f'S{session_n+1}'
        path = os.path.join(os.path.join(data_path, sub), sess) + f'/eeg/alldata_sbj{str(sub_n+1).zfill(2)}_sess{session_n+1}_{level}.set'
        # Read the epoched data with MNE
        epochs = mne.io.read_epochs_eeglab(path, verbose=False)
        # You could add some pre-processing here with MNE
        # We will just select some channels (mostly frontal ones)
        if (narrow_feature_space):
            epochs = epochs.drop_channels(list(set(epochs.ch_names) - set(ch_slice)))

        # Get the data and concatenante with others MATB levels
        tmp = epochs.get_data()
        epochs_data.extend(tmp)
        labels.extend([lab_idx]*len(tmp))
    print(len(labels))

    epochs_data = np.array(epochs_data)
    print(epochs_data.shape)
    labels = np.array(labels)

    # Train the model on all epochs from session 1 and 2
#     lr = LogisticRegression(C=1/10.)
#     clf = make_pipeline(pyriemann.estimation.Covariances(estimator='oas'),
#         pyriemann.classification.TSclassifier(clf=lr))
    clf = make_pipeline(
        ColumnConcatenator(), TSFreshFeatureExtractor(n_jobs=-1, show_warnings=False), RandomForestClassifier(n_jobs=-1,random_state=0)      
        )
    clf.fit(epochs_data, labels)

    #Validate on Session 3
    print(f"{sub_n} - session 2")
    session_n = 2
    epochs_data = []
    # labels = []
    # for lab_idx, level in enumerate(diff_submission):
    sub = 'P{0:02d}'.format(sub_n+1)
    sess = f'S{session_n+1}'
    path = os.path.join(os.path.join(data_path, sub), sess) + f'/eeg/testset_sbj{str(sub_n+1).zfill(2)}_sess{session_n+1}.set'
    # print(path)
    # # Read the epoched data with MNE
    epochs = mne.io.read_epochs_eeglab(path, verbose=False)
    # You could add some pre-processing here with MNE
    # We will just select some channels (mostly frontal ones)
    if (narrow_feature_space):
        epochs = epochs.drop_channels(list(set(epochs.ch_names) - set(ch_slice)))

    # Get the data and concatenante with others MATB levels
    tmp = epochs.get_data()
    # print(type(tmp))
    epochs_data.extend(tmp)
    # labels.extend(len(tmp))

    epochs_data = np.array(epochs_data)
    print(epochs_data.shape)
    # labels = np.array(labels)

    # Use trained model to predict for all epochs of session 2 and compute accuracy
    y_pred = clf.predict(epochs_data)
    # print(y_pred)
    print(len(y_pred))

    submission = pd.DataFrame({'epochID':np.arange(len(y_pred)), f'prediction_sub{sub_n}' : y_pred})
    submission.to_csv(f"{file_time}_{sub_n}_submission.csv",header=True,index=False)

    filename = f"{file_time}_{sub_n}_finalized_model.sav"
    pickle.dump(clf, open(filename, 'wb'))
    all_results =  all_results.merge(submission,left_on='epochID', right_on='epochID')

    
all_results.to_csv(f"{file_time}_allresults_submission.csv",header=True,index=False)


