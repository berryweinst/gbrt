from data_utils import parse_data
from gbrt_algorithm import gbrt
from feature_selection import ensemble_feature_importance
import operator
import pandas as pd
import argparse


class HParams(object):
    def __init__(self, num_trees=50, max_depth=3, min_node_size=0, weight_decay=1.0, sub_samp=1.0, num_thresholds=10,
                 verbose=1):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.weight_decay = weight_decay
        self.sub_samp = sub_samp
        self.num_thresholds = num_thresholds
        self.verbose = verbose


parser = argparse.ArgumentParser()

# example: python main.py -nt 300 -md 4 -mns 0 -wd 0.01 -ss 0.7 -v 0 -nthreh 10
parser.add_argument("-nt", "--num_trees", help="Number of trees", type=int)
parser.add_argument("-md", "--max_depth", help="Max trees depth", type=int)
parser.add_argument("-mns", "--min_node_size", help="Min Node Size", type=int)
parser.add_argument("-wd", "--weight_decay", help="Learning Rate/Weight decay", type=float)
parser.add_argument("-ss", "--sub_samp", help="sub sampling fraction", type=float)
parser.add_argument("-v", "--verbose", help="verbose", type=int)
parser.add_argument("-nthreh", "--num_threshold", help="num Thresholds", type=int)

args = parser.parse_args()

train_dataset, test_dataset = parse_data('data/train.csv')
params = HParams(num_trees=args.num_trees, max_depth=args.max_depth, min_node_size=args.min_node_size,
                 weight_decay=args.weight_decay, sub_samp=args.sub_samp, verbose=args.verbose,
                 num_thresholds=args.num_threshold)
model, logs = gbrt(train_data=train_dataset.data, test_data=test_dataset.data, label_name=train_dataset.label_name, params=params)
# ToDo - save logs to file

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
