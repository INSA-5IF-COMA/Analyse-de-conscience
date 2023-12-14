from test import explain_temporal
import numpy as np

subject = 3

# 0: SelfStim
# 1: CtrlStim
# 2: SelfRest
# 3: CtrlRest
# 4: SelfSoc
# 5: CtrlSoc

print("SelfStim vs SelfSoc")
grouped_ts_1_5r = explain_temporal(f"conf9003d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [0,4], [0,1], subject, 1)

print("SelfStim vs CtrlSoc")
grouped_ts_1_6r = explain_temporal(f"conf9004d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [0,5], [0,1], subject, 1)

print("CtrlStim vs SelfSoc")
grouped_ts_2_5r = explain_temporal(f"conf9007d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [1,4], [0,1], subject, 1)

print("CtrlStim vs CtrlSoc")
grouped_ts_2_6r = explain_temporal(f"conf9008d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [1,5], [0,1], subject, 1)

print("SelfRest vs SelfSoc")
grouped_ts_3_5r = explain_temporal(f"conf9010d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [2,4], [0,1], subject, 1)

print("SelfRest vs CtrlSoc")
grouped_ts_3_6r = explain_temporal(f"conf9011d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [2,5], [0,1], subject, 1)

print("CtrlRest vs SelfSoc")
grouped_ts_4_5r = explain_temporal(f"conf9012d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [3,4], [0,1], subject, 1)

print("CtrlRest vs CtrlSoc")
grouped_ts_4_6r = explain_temporal(f"conf9013d/checkpoint_{subject:02d}_00.ckpt", "/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv", [3,5], [0,1], subject, 1)


# Stim
#minlength = np.min((grouped_ts_1_3.shape[1], grouped_ts_1_4.shape[1], grouped_ts_2_3.shape[1], grouped_ts_2_4.shape[1]))
#sit_stim = np.mean((grouped_ts_1_3[:,:minlength], grouped_ts_1_4[:,:minlength], grouped_ts_2_3[:,:minlength], grouped_ts_2_4[:,:minlength]), axis=0)
#
# resample at 0.5s
#sit_stim_res = np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[0])
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[1])))
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[2])))
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[3])))
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[4])))
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[5])))
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[6])))
#sit_stim_res = np.vstack((sit_stim_res, np.interp(np.arange(0,sit_stim.shape[1], 10), np.arange(0,sit_stim.shape[1]), sit_stim[7])))


# Soc
minlength = np.min((grouped_ts_1_5r.shape[1], grouped_ts_2_5r.shape[1], grouped_ts_3_5r.shape[1], grouped_ts_4_5r.shape[1], grouped_ts_1_6r.shape[1], grouped_ts_2_6r.shape[1], grouped_ts_3_6r.shape[1], grouped_ts_4_6r.shape[1]))
sit_soc = np.mean((grouped_ts_1_5r[:,:minlength], grouped_ts_2_5r[:,:minlength], grouped_ts_3_5r[:,:minlength], grouped_ts_4_5r[:,:minlength], grouped_ts_1_6r[:,:minlength], grouped_ts_2_6r[:,:minlength], grouped_ts_3_6r[:,:minlength], grouped_ts_4_6r[:,:minlength]), axis=0)

# resample at 0.5s
sit_soc_res = np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[0])
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[1])))
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[2])))
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[3])))
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[4])))
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[5])))
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[6])))
sit_soc_res = np.vstack((sit_soc_res, np.interp(np.arange(0,sit_soc.shape[1], 10), np.arange(0,sit_soc.shape[1]), sit_soc[7])))


# Self
#minlength = np.min((grouped_ts_1_2r.shape[1], grouped_ts_3_4r.shape[1], grouped_ts_5_6r.shape[1]))
#sit_self = np.mean((grouped_ts_1_2r[:,:minlength], grouped_ts_3_4r[:,:minlength], grouped_ts_5_6r[:,:minlength]), axis=0)

# resample at 0.5s
#sit_self_res = np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[0])
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[1])))
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[2])))
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[3])))
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[4])))
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[5])))
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[6])))
#sit_self_res = np.vstack((sit_self_res, np.interp(np.arange(0,sit_self.shape[1], 10), np.arange(0,sit_self.shape[1]), sit_self[7])))



