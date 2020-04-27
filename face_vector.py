import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, LabelEncoder

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


def regression_model(h_layers, h_units):
    model = Sequential()
    model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu'))

    for h in range(0, h_layers):
        model.add(Dense(h_units, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='linear'))

    opt = SGD(lr=0.01, momentum=0.9)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    return model


def objective(trial):
    # hyperparameter setting
    regression_method = trial.suggest_categorical('regression_method', (
        'ridge', 'lasso', 'SVC', 'RandomForest',
        'REG'
    ))
    if regression_method == 'ridge':
        ridge_alpha = trial.suggest_uniform('ridge_alpha', 0.0, 2.0)
        model = sklearn.linear_model.Ridge(alpha=ridge_alpha)
    elif regression_method == 'lasso':
        lasso_alpha = trial.suggest_uniform('lasso_alpha', 0.0, 2.0)
        model = sklearn.linear_model.Lasso(alpha=lasso_alpha)
    elif regression_method == 'SVC':
        svc_c = trial.suggest_loguniform("svc_c", 1e-10, 1e10)
        model = sklearn.svm.SVC(C=svc_c, gamma="auto")
    elif regression_method == 'RandomForest':
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        model = RandomForestClassifier(max_depth=rf_max_depth)
    else:
        h_layers = trial.suggest_int('h_layers', 1, 5)
        h_units = trial.suggest_int('h_units', 1, 50)
        model = regression_model(h_layers=h_layers, h_units=h_units)

    X_train, y_train, X_val, y_val, x_test, Y_test = getdata()

    # model training and evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    # output: evaluation score
    return error


neptune.init(project_qualified_name='4ND4/sandbox')
neptune.create_experiment(name='optuna sweep')

monitor = opt_utils.NeptuneMonitor()

study = optuna.create_study(direction='minimize')
study.optimize(
    objective,
    n_trials=100,
    callbacks=[monitor]
)

print('Minimum mean squared error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
