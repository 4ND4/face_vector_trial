import keras
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
from optuna.samplers import TPESampler
from sklearn.preprocessing import Normalizer, LabelEncoder

maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5
optimizer_direction = 'minimize'
number_of_random_points = 25  # random searches to start opt process
maximum_time = 4 * 60 * 60  # seconds
results_directory = 'output/'

num_classes = 1
VECTOR_SIZE = 512
FACE_DETECTION = False
channel = 3


class Objective(object):
    def __init__(self, xcalib, ycalib, xvalid, yvalid, dir_save,
                 max_epochs, early_stop, learn_rate_epochs,
                 input_shape, number_of_classes):
        self.xcalib = xcalib
        self.ycalib = ycalib
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes
        self.xvalid = xvalid
        self.yvalid = yvalid

    def __call__(self, trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)
        learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.00025)
        freeze_layers = trial.suggest_categorical('freeze_layers', [15, 25, 32, 40, 100, 150])

        # implement resnet50

        resnet_50 = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
        )

        # start - changed

        #x = Flatten()(resnet_50.output)

        x = resnet_50.output

        # end - changed

        # introduced

        x = GlobalAveragePooling2D()(x)

        # introduced

        x = Dropout(drop_out)(x)

        predictions = Dense(1)(x)

        model = Model(inputs=resnet_50.inputs, outputs=predictions)

        # introduced

        for layer in model.layers[:-freeze_layers]:
            layer.trainable = False

        model.compile(loss='mse',
                      optimizer=optimizers.RMSprop(lr=learning_rate),
                      metrics=['mae'])
        model.summary()

        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_cnn.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=self.learn_rate_epochs,
                                            verbose=0, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)]

        # fit the model
        h = model.fit(x=self.xcalib, y=self.ycalib,
                      batch_size=batch_size,
                      epochs=self.max_epochs,
                      # validation_split=0.25,
                      validation_data=(self.xvalid, self.yvalid),
                      shuffle=True, verbose=0,
                      callbacks=callbacks_list)

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


def getdata():
    _data = np.load('output/DeepUAge-faces-embeddings_{}_{}.npz'.format(VECTOR_SIZE, FACE_DETECTION))
    embeddings_val_npz_path = 'output/DeepUAge-val-faces-embeddings-{}-{}.npz'.format(VECTOR_SIZE, FACE_DETECTION)

    trainX, trainy, valX, valy = _data['arr_0'], _data['arr_1'], _data['arr_2'], _data['arr_3']
    # print('Dataset: train=%d, test=%d' % (trainX.shape[0], valX.shape[0]))

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

    test_x, test_y = test_data['arr_0'], test_data['arr_1']

    # print('Dataset: test=%d' % (valX.shape[0]))

    # normalize input vectors

    test_x = in_encoder.transform(test_x)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(test_y)

    # modify so it complies with 4 dimensions.

    trainX = np.repeat(trainX[:, :, np.newaxis], 3, axis=2)
    valX = np.repeat(valX[:, :, np.newaxis], 3, axis=2)
    test_x = np.repeat(test_x[:, :, np.newaxis], 3, axis=2)

    trainX = np.repeat(trainX[:, :, np.newaxis], 32, axis=2)
    valX = np.repeat(valX[:, :, np.newaxis], 32, axis=2)
    test_x = np.repeat(test_x[:, :, np.newaxis], 32, axis=2)

    return trainX, trainy, valX, valy, test_x, test_y


train_X, train_Y, val_X, val_Y, _, _ = getdata()

# shape_of_input = (train_X.shape[0], 512, channel)
shape_of_input = (512, 32, channel)

neptune.init(project_qualified_name='4ND4/sandbox')
result = neptune.create_experiment(name='optuna Resnet50 Face Vectors')
monitor = opt_utils.NeptuneMonitor()
callback = [monitor]
n_trials = 100

objective = Objective(train_X, train_Y, val_X, val_Y, results_directory,
                      maximum_epochs, early_stop_epochs,
                      learning_rate_epochs, shape_of_input, num_classes)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction=optimizer_direction,
                            sampler=TPESampler(n_startup_trials=number_of_random_points))

study.optimize(
    objective,
    timeout=maximum_time,
    callbacks=callback
)

# save results
df_results = study.trials_dataframe()
df_results.to_pickle(results_directory + 'df_optuna_results.pkl')
df_results.to_csv(results_directory + 'df_optuna_results.csv')

print('Minimum error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
print('Best trial: ' + str(study.best_trial))
