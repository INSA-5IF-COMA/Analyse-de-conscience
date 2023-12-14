from tqdm import tqdm

model.train()
ig = IntegratedGradients(model)

ap_np = np.array(all_predicted)
features = featuresTest[targetsTest.detach().numpy()==ap_np]
targets = targetsTest[targetsTest.detach().numpy()==ap_np]

prevt = targets[0]
all_ts = []
all_ts_labels = []
final_ts = []
final_ts_labels = []
for ti in tqdm(range(len(targets))):
  t = targets[ti]
  if t!= prevt:
    avg_all_ts = np.zeros(features.shape[1] + (len(all_ts)-1) * test_overlap_size)
    avg_all_ts_weights = np.zeros(features.shape[1] + (len(all_ts)-1) * test_overlap_size)
    for i in range(len(all_ts)):
      start_idx = i*test_overlap_size
      end_idx = start_idx + features.shape[1]
      avg_all_ts[start_idx:end_idx] += all_ts[i]
      avg_all_ts_weights[start_idx:end_idx] += 1
    avg_all_ts /= avg_all_ts_weights
    final_ts.append(avg_all_ts)
    final_ts_labels.append(prevt)
    all_ts = []
    all_ts_labels = []

  attr, delta = ig.attribute(features[t].unsqueeze(0), target=targets[t].unsqueeze(0), return_convergence_delta=True)
  attr_np = attr.detach().cpu().numpy()
  attr_np = np.mean(np.clip(attr_np[0], a_min=0, a_max=None), axis=1)

  all_ts.append(attr_np)
  all_ts_labels.append(targets[t])
  
  prevt = t

avg_all_ts = np.zeros(features.shape[1] + (len(all_ts)-1) * test_overlap_size)
avg_all_ts_weights = np.zeros(features.shape[1] + (len(all_ts)-1) * test_overlap_size)
for i in range(len(all_ts)):
  start_idx = i*test_overlap_size
  end_idx = start_idx + features.shape[1]
  avg_all_ts[start_idx:end_idx] += all_ts[i]
  avg_all_ts_weights[start_idx:end_idx] += 1
avg_all_ts /= avg_all_ts_weights
final_ts.append(avg_all_ts)
final_ts_labels.append(prevt)
