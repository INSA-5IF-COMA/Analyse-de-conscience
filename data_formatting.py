import numpy as np
import pandas as pd
from sklearn import preprocessing

# split between train and test, all first event goes into test the rest in train
def split_train_test(targets_numpy, features_numpy, subjects, train_subjects, test_subjects):
    train_idx = np.bitwise_or.reduce(np.array([subjects==i for i in train_subjects]), axis=0)
    test_idx = np.bitwise_or.reduce(np.array([subjects==i for i in test_subjects]), axis=0)

    features_train = features_numpy[train_idx]
    targets_train = targets_numpy[train_idx]
    features_test = features_numpy[test_idx]
    targets_test = targets_numpy[test_idx]

    return features_train, targets_train, features_test, targets_test



def split_sequence(features, labels, window_size):
    """sépare les données (en format frame par frame) dans un tableau en 3d par séquence de window_size frame

    window_size : frame"""

    ret_features, ret_labels, seq_features, seq_labels = [], [], [], []
    framecount = 0
    last_event = labels[0]
    for i in range(len(features)):
        seq_features.append(features[i])
        # seq_labels.append(labels[i])

        if framecount == window_size - 1:
            ret_features.append(np.array(seq_features))
            if (labels[i] == 65) or (labels[i] == 66):
                ret_labels.append(0)
            if (labels[i] == 67) or (labels[i] == 68):
                ret_labels.append(1)
            if (labels[i] == 69) or (labels[i] == 70):
                ret_labels.append(2)
            seq_features = []
            seq_labels = []
            framecount = -1

        if labels[i] != last_event:
            last_event = labels[i]
            seq_features = []
            seq_labels = []
            framecount = -1
        framecount += 1

    ret_features = np.array(ret_features)
    ret_labels = np.array(ret_labels)
    return ret_features, ret_labels


def split_sequence_overlap(features, labels, window_size, step_size):
    ret_features, ret_labels = [], []
    for i in range(0, len(features)-window_size+1, step_size):
        if labels[i]==labels[i+window_size-1]:
            ret_features.append(features[i:i+window_size])
            ret_labels.append(labels[i])
    return np.array(ret_features), np.array(ret_labels)

def split_sequence_nooverlap(features, labels, window_size, step_size):
    ret_features, ret_labels = [], []
    start_idx=0
    for i in range(1, len(features)):
        if labels[i]!=labels[i-1]:
          ret_features.append(features[start_idx:i])
          ret_labels.append(labels[i-1])
          start_idx=i
    ret_features.append(features[start_idx:len(features)])
    ret_labels.append(labels[len(features)-1])
    return np.array(ret_features, dtype=object), np.array(ret_labels)


def normalize_data(train_df, normalise_individual_subjects):
  subjects = pd.factorize(train_df['Subject'])[0]
  train_df = train_df.drop(['Condition','Subject','Emotion','Presence'], axis= 1)
  features_numpy = train_df.to_numpy(dtype='float32')

  unique_values = np.array([len(np.unique(features_numpy[:,i])) for i in range(features_numpy.shape[1])])
  floatcols = unique_values!=2  # retrieve all columns that are not boolean
  if normalise_individual_subjects:
    for i in np.unique(subjects): # normalise each subject individually
      scaler = preprocessing.StandardScaler()  # BEST
      (features_numpy[(subjects==i)])[:, floatcols] = scaler.fit_transform((features_numpy[(subjects==i)])[:, floatcols])
  else:
    scaler = preprocessing.StandardScaler()  # BEST
    features_numpy[:, floatcols] = scaler.fit_transform(features_numpy[:, floatcols])
  return features_numpy


def set_targets(train_df, list_targets, list_labels):

  nclasses = len(list_targets)

  for i in range(6):
    if i not in list_targets:
      train_df = train_df.drop(train_df[train_df['Condition'] == (65+i)].index) 

  targets_numpy = train_df.Condition.values
  for i in range(len(list_targets)):
    targets_numpy[targets_numpy==(65+list_targets[i])] = list_labels[i] 
    
  return train_df, nclasses, targets_numpy