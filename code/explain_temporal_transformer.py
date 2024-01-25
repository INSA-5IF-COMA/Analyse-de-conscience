import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression as LR
from data_formatting import split_sequence_overlap, split_sequence_nooverlap, split_sequence, split_train_test, normalize_data, set_targets
import parameters
from tqdm import tqdm
from captum.attr import IntegratedGradients
from transformer_architecture import Transformer


# explain_idx : choose which class to explain (class labels 0,1)
# why test_seq_dim is 900 ? -> changed it to 900 so code works
def explain_temporal(model_filename, train_data, list_targets, list_labels, test_subj, explain_idx, test_overlap_size=100, test_seq_dim = 1000):

  parameters.initialize_parameters()


  print(torch.__version__)
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # override
  device = 'cpu'

  print(f"Using {device}")


  points_face_contour = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
  points_limbs = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
  points_nose = (39, 40, 41, 42, 43, 44, 45, 46, 47)
  points_mouth = (60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76)
  points_eyes = (29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 80)
  au_eyes = (83, 84, 85, 86, 87, 88, 99)
  au_mouth = (90, 91, 92, 93, 94, 95, 96, 97, 98)
  au_nose = (89)



  # Get data
  train_df = pd.read_csv(train_data,  delimiter=",")  # 101 features (only AU_r) (filtered)

  train_df, nclasses, targets_numpy = set_targets(train_df, list_targets, list_labels)

  # Convert the subject names (strings) into numbers
  subjects = pd.factorize(train_df['Subject'])[0]


  # normalise the features
  features_numpy = normalize_data(train_df, parameters.normalise_individual_subjects)
  input_dim = features_numpy.shape[1]
  print(f"Number of features: {input_dim}")


  subj = np.unique(subjects)

  test_idx = np.array([test_subj])
  trainval_idx = np.delete(subj, np.where(subj==test_subj)) # take out test subject from trainval

  val_idx = np.array([test_subj+1, test_subj+2, test_subj+3, test_subj+4]) # use four following subjects for validation
  val_idx = val_idx%len(subj)

  train_idx = np.setxor1d(subj, test_idx)
  train_idx = np.setxor1d(train_idx, val_idx)

  print("Generating train/val/test split...")
  features_train, targets_train, features_val, targets_val, features_test, targets_test = split_train_test(targets_numpy, features_numpy, subjects, train_idx, val_idx, test_idx)

  print("Generating sequences...")
  original_targets = targets_test
  if parameters.test_with_subsequences:
    features_test, targets_test = split_sequence_overlap(features_test, targets_test, test_seq_dim, test_overlap_size)
  else:
    features_test, targets_test = split_sequence_nooverlap(features_test, targets_test, test_seq_dim, test_overlap_size)

  print(f"Number of test examples: {len(targets_test)}")


  # create feature and targets tensor for test set.
  if parameters.test_with_subsequences:
      featuresTest = torch.from_numpy(features_test)
      targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # data type is long
      test = TensorDataset(featuresTest, targetsTest)
      test_loader = DataLoader(test, batch_size=parameters.batch_size, shuffle=False)

  # validation
  features_val, targets_val = split_sequence_overlap(features_val, targets_val, parameters.seq_dim, parameters.overlap_size)
  featuresVal = torch.from_numpy(features_val)
  targetsVal = torch.from_numpy(targets_val).type(torch.LongTensor)  # data type is long
  val = TensorDataset(featuresVal, targetsVal)
  val_loader = DataLoader(val, batch_size=parameters.batch_size, shuffle=False)


  print("Loading model.")
  learning_rate = 0.0001 # Adam
  weight_decay = 10e-4

  # model = CNN(input_dim, parameters.seq_dim).to(device)
  model = Transformer(parameters.batch_size, 64).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  # Cross Entropy Loss
  error = nn.CrossEntropyLoss()
  error_cpu = nn.CrossEntropyLoss().to('cpu')



  # loss_list = []
  # val_loss_list = []
  # iteration_list = []
  # accuracy_list = []
  # val_accuracy_list = []
  count = 0
  # min_val_loss = 1e6
  # max_val_accuracy = 0
  # clip_value=0.5


  # Test
  model.load_state_dict(torch.load(model_filename, map_location=device))
  test_model = model


  correct = 0
  total = 0
  prev_label = -1
  class_hist = np.zeros(nclasses, dtype='int')
  all_predicted = []
  all_labels = []
  all_outputs = np.empty((0, nclasses), dtype='float')

  # Iterate through test dataset
  test_model.eval()
  with torch.no_grad():
    if parameters.test_with_subsequences:
      for features, labels in test_loader:
        features = Variable(features.view(-1, test_seq_dim, input_dim)).to(device)

        model.eval()
        # Forward propagation
        outputs = test_model(features)

        test_loss = error(outputs.to('cpu'), labels)
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        predicted = predicted.to('cpu')
        if parameters.test_use_max:
          bi = 0
          for l in labels:
            if l!=prev_label and prev_label!=-1:
              final_predicted = np.argmax(class_hist)
              if final_predicted==prev_label:
                correct += 1
              class_hist = np.zeros(nclasses, dtype='int')
              total += 1
              all_predicted.append(final_predicted)
              all_labels.append(l)

            class_hist[predicted[bi]] += 1
            prev_label = l
            bi += 1
        else:
          # Total number of labels
          total += labels.size(0)
          correct += (predicted == labels).sum()
          all_predicted.extend(list(predicted.detach().numpy()))
          all_labels.extend(list(labels.detach().numpy()))
          all_outputs = np.concatenate((all_outputs, outputs.data.to('cpu').reshape(-1, nclasses)))

      if parameters.test_use_max and np.sum(class_hist)>0:
        final_predicted = np.argmax(class_hist)
        if final_predicted==prev_label:
          correct += 1
        total += 1
        all_predicted.append(final_predicted)
        all_labels.append(l)

      print("Running Integrated Gradients.")
      ap_np = np.array(all_predicted)
      t_np = targetsTest.detach().numpy()
      features = featuresTest.to(device)
      targets = targetsTest.to(device)

      model.train()
      ig = IntegratedGradients(model)

      prevt = targets[0]
      all_ts = []
      all_ts_labels = []
      all_ts_valid = []  # True if sequence was classified correctly, False otherwise
      final_ts = []
      final_ts_labels = []
      for ti in tqdm(range(len(targets))):
        t = targets[ti]
        if t!= prevt: # if condition/label changes compute average feature attribution over all overlapping time windows
          avg_all_ts = np.zeros((features.shape[1] + (len(all_ts)-1) * test_overlap_size, features.shape[2]))
          avg_all_ts_weights = np.zeros((features.shape[1] + (len(all_ts)-1) * test_overlap_size, features.shape[2]))
          for i in range(len(all_ts)):
            if all_ts_valid[i]:
              start_idx = i*test_overlap_size
              end_idx = start_idx + features.shape[1]
              avg_all_ts[start_idx:end_idx] += all_ts[i]
              avg_all_ts_weights[start_idx:end_idx] += 1
          if not (avg_all_ts_weights==0).any():
            avg_all_ts /= avg_all_ts_weights
          final_ts.append(avg_all_ts)
          final_ts_labels.append(prevt)
          all_ts = []
          all_ts_labels = []
          all_ts_valid = [] 

        attr, delta = ig.attribute(features[ti].unsqueeze(0), target=targets[ti].unsqueeze(0), return_convergence_delta=True)
        attr_np = attr.detach().cpu().numpy()
        attr_np = np.clip(attr_np[0], a_min=0, a_max=None)  # clip negative attributions

        all_ts.append(attr_np)
        all_ts_labels.append(targets[t])
        all_ts_valid.append(ap_np[t]==t_np[t])
        
        prevt = t

      # run once more for last condition/label
      avg_all_ts = np.zeros((features.shape[1] + (len(all_ts)-1) * test_overlap_size, features.shape[2]))
      avg_all_ts_weights = np.zeros((features.shape[1] + (len(all_ts)-1) * test_overlap_size, features.shape[2]))
      for i in range(len(all_ts)):
        if all_ts_valid[i]:
          start_idx = i*test_overlap_size
          end_idx = start_idx + features.shape[1]
          avg_all_ts[start_idx:end_idx] += all_ts[i]
          avg_all_ts_weights[start_idx:end_idx] += 1
      if not (avg_all_ts_weights==0).any():
        avg_all_ts /= avg_all_ts_weights
      final_ts.append(avg_all_ts)
      final_ts_labels.append(prevt)

      # group feature attributions according to semantic groups
      grouped_ts = np.mean(final_ts[explain_idx][:,points_limbs], axis=1)
      grouped_ts = np.vstack((grouped_ts, np.mean(final_ts[explain_idx][:,points_face_contour], axis=1)))
      grouped_ts = np.vstack((grouped_ts, np.mean(final_ts[explain_idx][:,points_eyes], axis=1)))
      grouped_ts = np.vstack((grouped_ts, np.mean(final_ts[explain_idx][:,points_nose], axis=1)))
      grouped_ts = np.vstack((grouped_ts, np.mean(final_ts[explain_idx][:,points_mouth], axis=1)))
      grouped_ts = np.vstack((grouped_ts, np.mean(final_ts[explain_idx][:,au_eyes], axis=1)))
      grouped_ts = np.vstack((grouped_ts, final_ts[explain_idx][:,au_nose]))
      grouped_ts = np.vstack((grouped_ts, np.mean(final_ts[explain_idx][:,au_mouth], axis=1)))
    
    else:
      count=0
      for features in features_test:
        features = torch.tensor(features)
        features = torch.unsqueeze(features, 0).to(device)
        labels = torch.unsqueeze(torch.tensor(targets_test[count]), 0)

        # Forward propagation
        outputs = test_model(features)

        test_loss = error(outputs.to('cpu'), labels)
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        predicted = predicted.to('cpu')

        # Total number of labels
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count += 1
      grouped_ts = 0 # not yet implemented

  accuracy = correct / float(total)
  print(f"Test accuracy: {accuracy}")

  return grouped_ts
