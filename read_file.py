import numpy as np
import utils


pre = np.load('checkpoints/Supervised_small_RUL_from1_CNC_FT_bs16_lr0.0003_seed42_20250815_215452/test_preds.npy')
trg = np.load('checkpoints/Supervised_small_RUL_from1_CNC_FT_bs16_lr0.0003_seed42_20250815_215452/test_targets.npy')

print(pre)
print(trg)

score = utils.scoring_function_v2(pre,trg)
print(score)