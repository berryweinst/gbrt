from data_utils import parse_data
from gbrt_algorithm import gbrt
from feature_selection import ensemble_feature_importance
import operator
import pandas as pd


class hparams(object):
    def __init__(self, num_trees=50, max_depth=3, min_node_size=0, weight_decay=1.0, sub_samp=1.0, num_thresholds=3,
                 verbose=True):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.weight_decay = weight_decay
        self.sub_samp = sub_samp
        self.num_thresholds = num_thresholds
        self.verbose = verbose


train_dataset, test_dataset = parse_data('data/train.csv')
params = hparams(num_trees=300, max_depth=2, min_node_size=0, weight_decay=0.05, sub_samp=0.4, verbose=False)
model = gbrt(train_data=train_dataset.data, test_data=test_dataset.data, label_name=train_dataset.label_name, params=params)

features_dict = ensemble_feature_importance(test_dataset.data, train_dataset.label_name, model)
n = 5
print("Most %d important features are:" % (n))
for f in sorted(features_dict.items(), key=operator.itemgetter(1), reverse=True)[0:n]:
    print("feature: %s, score: %.3f" % (f[0], f[1]))


# Test data for kaggle
test_kaggle = parse_data('data/test.csv', for_train=False, train_dataset=train_dataset)
test_kaggle_pred = test_kaggle.data.apply(lambda xi: model.evaluate(xi[:], params.num_trees), axis=1)
df_test_kaggle_pred = pd.DataFrame({'Id': test_kaggle_pred.index, 'SalePrice': test_kaggle_pred.values})
df_test_kaggle_pred.to_csv('data/pred_kaggle.csv', index=False)
