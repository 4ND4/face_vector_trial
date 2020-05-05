import keras
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from keras import Model, Input, Sequential, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import BatchNormalization, Activation, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import numpy as np
from optuna.samplers import TPESampler
from sklearn.preprocessing import Normalizer, LabelEncoder


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
        num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 2, 4)
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        num_dense_nodes = trial.suggest_categorical('num_dense_nodes',
                                                    [64, 128, 512, 1024])
        dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor',
                                                        [2, 4, 8])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)

        learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.00025)

        dict_params = {'num_cnn_blocks': num_cnn_blocks,
                       'num_filters': num_filters,
                       'kernel_size': kernel_size,
                       'num_dense_nodes': num_dense_nodes,
                       'dense_nodes_divisor': dense_nodes_divisor,
                       'batch_size': batch_size,
                       'drop_out': drop_out,
                       'learning_rate': learning_rate
                       }


        # start of cnn coding
        #input_tensor = Input(shape=self.input_shape)

        # implement resnet50

        resnet_50 = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            #input_tensor=input_tensor,
            input_shape=self.input_shape,
            # pooling=None,
            # classes=1
        )

        x = Flatten()(resnet_50.output)
        x = Dense(1)(x)
        model = Model(inputs=resnet_50.inputs, outputs=x)

        model.compile(loss='mse',
                      optimizer=optimizers.RMSprop(lr=2e-5),
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
                      batch_size=dict_params['batch_size'],
                      epochs=self.max_epochs,
                      # validation_split=0.25,
                      validation_data=(self.xvalid, self.yvalid),
                      shuffle=True, verbose=0,
                      callbacks=callbacks_list)

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


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

# shape_of_input = (512, 1, 1)


width = VECTOR_SIZE
height = 1
channel = 1


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

    return trainX, trainy, valX, valy, test_x, test_y


train_X, train_Y, val_X, val_Y, _, _ = getdata()


img_rows = 512
img_cols = 1


train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
val_X = val_X.reshape(val_X.shape[0], img_rows, img_cols, 1)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

shape_of_input = (VECTOR_SIZE, 1, channel)

#neptune.init(project_qualified_name='4ND4/sandbox')
# result = neptune.create_experiment(name='optuna CNN')

# monitor = opt_utils.NeptuneMonitor()
# callback = [monitor]
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
    # callbacks=callback
)

# save results
df_results = study.trials_dataframe()
df_results.to_pickle(results_directory + 'df_optuna_results.pkl')
df_results.to_csv(results_directory + 'df_optuna_results.csv')

print('Minimum error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))
print('Best trial: ' + str(study.best_trial))
