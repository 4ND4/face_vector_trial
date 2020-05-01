import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
import xgboost as xgb

FACE_DETECTION = False
VECTOR_SIZE = 512


def getdata():
    _data = np.load('output/DeepUAge-faces-embeddings_{}_{}.npz'.format(VECTOR_SIZE, FACE_DETECTION))
    embeddings_val_npz_path = 'output/DeepUAge-val-faces-embeddings-{}-{}.npz'.format(VECTOR_SIZE, FACE_DETECTION)

    trainX, trainy, valX, valy = _data['arr_0'], _data['arr_1'], _data['arr_2'], _data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], valX.shape[0]))

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    valX = in_encoder.transform(valX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    valy = out_encoder.transform(valy)

    # get test set

    test_data = np.load(embeddings_val_npz_path)

    testX, testY = test_data['arr_0'], test_data['arr_1']

    print('Dataset: test=%d' % (valX.shape[0]))

    # normalize input vectors

    testX = in_encoder.transform(testX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(testY)

    return trainX, trainy, valX, valy, testX, testY


def objective(trial):
    #(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    #train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)

    train_x, train_y, val_x, val_y, _, _ = getdata()

    dtrain = xgb.DMatrix(train_x, label=train_y)
    #dtest = xgb.DMatrix(test_x, label=test_y)
    dtest = xgb.DMatrix(val_x, label=val_y)

    param = {
        "silent": 1,
        #"objective": "binary:logistic",
        #"objective": "reg:squaredlogerror",
        #"objective": "reg:logistic",
        "eval_metric": "mae",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dtest)
    pred_labels = np.rint(preds)

    mae = sklearn.metrics.mean_absolute_error(val_y, pred_labels)
    return mae


neptune.init(project_qualified_name='4ND4/sandbox')
neptune.create_experiment(name='optuna sweep')

monitor = opt_utils.NeptuneMonitor()

study = optuna.create_study(direction='minimize')
study.optimize(
    objective,
    n_trials=100,
    callbacks=[monitor]
)

print('Minimum mean absolute error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
print('Best trial: ' + str(study.best_trial))
