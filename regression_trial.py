import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import Normalizer, LabelEncoder

FACE_DETECTION = False
VECTOR_SIZE = 512
EPOCHS = 100

log_report = False


def getdata():
    _data = np.load('output/DeepUAge-faces-embeddings_{}_{}.npz'.format(VECTOR_SIZE, FACE_DETECTION))
    embeddings_val_npz_path = 'output/DeepUAge-val-faces-embeddings-{}-{}.npz'.format(VECTOR_SIZE, FACE_DETECTION)

    trainX, trainy, valX, valy = _data['arr_0'], _data['arr_1'], _data['arr_2'], _data['arr_3']

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

    # normalize input vectors

    testX = in_encoder.transform(testX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(testY)

    return trainX, trainy, valX, valy, testX, testY


def regression_model(h_layers, h_units, lr):
    model = Sequential()
    model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu'))

    for h in range(0, h_layers):
        model.add(Dense(h_units, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='linear'))

    # opt = SGD(lr=lr, momentum=0.9)

    opt = Adam(lr=lr)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])
    return model


def objective(trial):
    # hyperparameter setting

    h_layers = trial.suggest_int('h_layers', 1, 50)
    h_units = trial.suggest_int('h_units', 1, 1024)
    lr = trial.suggest_loguniform("lr", 1e-2, 1e-1)
    model = regression_model(h_layers=h_layers, h_units=h_units, lr=lr)

    X_train, y_train, X_val, y_val, x_test, Y_test = getdata()

    # model training and evaluation

    es = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_mae', mode='max', verbose=1, save_best_only=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=64, epochs=EPOCHS, callbacks=[es, mc])
    y_pred = model.predict(X_val)

    if not np.any(np.isnan(y_pred)):

        error = sklearn.metrics.mean_absolute_error(y_pred, y_val)

    # output: evaluation score
        return error
    else:
        print('nan values')


callback = None
n_trials = 100

if log_report:
    neptune.init(project_qualified_name='4ND4/sandbox')
    neptune.create_experiment(name='optuna sweep')
    monitor = opt_utils.NeptuneMonitor()
    callback = [monitor]
    n_trials = 1

study = optuna.create_study(direction='minimize')
study.optimize(
    objective,
    n_trials=n_trials,
    callbacks=callback
)

print('Minimum mean absolute error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
