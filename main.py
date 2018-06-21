from data_utils import parse_data
from gbrt_algorithm import gbrt
from feature_selection import ensemble_feature_importance
import operator
import pandas as pd
import numpy as np
import argparse
import time



class HParams(object):
    def __init__(self, num_trees=50, max_depth=3, min_node_size=0, weight_decay=1.0, sub_samp=1.0, num_thresholds=10,
                 lr_step=0.5, lr_update=100, verbose=1):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.weight_decay = weight_decay
        self.sub_samp = sub_samp
        self.num_thresholds = num_thresholds
        self.lr_step = lr_step
        self.lr_update = lr_update
        self.verbose = verbose


parser = argparse.ArgumentParser()

# example: python main.py -nt 300 -md 4 -mns 0 -wd 0.01 -ss 0.7 -v 0 -nthreh 10
parser.add_argument("-nt", "--num_trees", help="Number of trees", type=int)
parser.add_argument("-md", "--max_depth", help="Max trees depth", type=int)
parser.add_argument("-mns", "--min_node_size", help="Min Node Size", type=int)
parser.add_argument("-wd", "--weight_decay", help="Learning Rate/Weight decay", type=float)
parser.add_argument("-ss", "--sub_samp", help="sub sampling fraction", type=float)
parser.add_argument("-nthreh", "--num_threshold", help="num Thresholds", type=int)
parser.add_argument("-lrs", "--lr_step", help="lr rate decay", type=float)
parser.add_argument("-lru", "--lr_update", help="lr update num iterations ", type=int)
parser.add_argument("-v", "--verbose", help="verbose", type=int)

args = parser.parse_args()

train_dataset, test_dataset = parse_data('data/train.csv')
params = HParams(num_trees=args.num_trees, max_depth=args.max_depth, min_node_size=args.min_node_size,
                 weight_decay=args.weight_decay, sub_samp=args.sub_samp,
                 num_thresholds=args.num_threshold, verbose=args.verbose)


# Training
model, logs = gbrt(train_data=train_dataset.data, test_data=test_dataset.data, label_name=train_dataset.label_name, params=params)

# Save model hyper params and training progress of the loss
logs['Hparams'] = HParams
np.save('nt_%d__md_%d__mns_%d__wd_%d__ss_%d__nthr_%d.npy' % \
        (args.num_trees, args.max_depth, args.min_node_size, args.weight_decay, args.sub_samp, args.num_threshold), logs)


# Test feature importace selection
features_dict = ensemble_feature_importance(test_dataset.data, train_dataset.label_name, model)
n = 5
print("Most %d important features are:" % n)
for f in sorted(features_dict.items(), key=operator.itemgetter(1), reverse=True)[0:n]:
    print("feature: %s, score: %.3f" % (f[0], f[1]))


# Test data for kaggle
test_kaggle = parse_data('data/test.csv', for_train=False, train_dataset=train_dataset)
test_kaggle_pred = test_kaggle.data.apply(lambda xi: model.evaluate(xi[:], params.num_trees), axis=1)
df_test_kaggle_pred = pd.DataFrame({'Id': test_kaggle_pred.index, 'SalePrice': test_kaggle_pred.values})
df_test_kaggle_pred.to_csv('data/pred_kaggle.csv', index=False)


#Save model
import pickle
pickle.dump(model, open('model_save__nt_%d__md_%d__mns_%d__wd_%d__ss_%d__nthr_%d.pkl' % \
                        (args.num_trees, args.max_depth, args.min_node_size, args.weight_decay, args.sub_samp, args.num_threshold), 'wb'))
