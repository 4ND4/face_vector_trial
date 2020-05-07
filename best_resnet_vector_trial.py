# imports
import keras
import neptune
import neptune_tensorboard as neptune_tb
import numpy as np
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, Normalizer

DEBUG = False
LOG_NEPTUNE = True
EPOCHS = 1000

if DEBUG:
    EPOCHS = 100

# parameters

VECTOR_SIZE = 512
FACE_DETECTION = False

PARAMS = {
    'epoch_nr': EPOCHS,
    'batch_size': 64,
    'learning_rate': 0.006,
    'input_shape': (512, 32, 3),
    'early_stop': 20
}

# start experiment

name = 'resnet50-experiment'

if LOG_NEPTUNE:
    neptune.init(project_qualified_name='4ND4/sandbox')
    neptune_tb.integrate_with_keras()
    result = neptune.create_experiment(name=name, params=PARAMS)

    name = result.id


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


# start of ResNet50 coding

resnet_50 = keras.applications.resnet.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=PARAMS.get('input_shape'),
)

x = Flatten()(resnet_50.output)

predictions = Dense(1)(x)

model = Model(inputs=resnet_50.inputs, outputs=predictions)

model.compile(loss='mse',
              optimizer=Adam(lr=PARAMS.get('learning_rate')),
              metrics=['mae'])

model.summary()

# callbacks for early stopping and for learning rate reducer

callbacks_list = [EarlyStopping(monitor='val_loss', patience=PARAMS.get('early_stop')),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                    patience=PARAMS.get('early_stop'),
                                    verbose=1, mode='auto', min_lr=PARAMS.get('learning_rate')),
                  ModelCheckpoint(filepath='checkpoint/model-{epoch:03d}-{val_loss:03f}.h5',
                                  monitor='val_loss', save_best_only=True)]

train_X, train_Y, val_X, val_Y, test_X, test_Y = getdata()


# fit the model
h = model.fit(x=train_X, y=train_Y,
              batch_size=PARAMS.get('batch_size'),
              epochs=PARAMS.get('epoch_nr'),
              validation_data=(val_X, val_Y),
              shuffle=True, verbose=1,
              callbacks=callbacks_list)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(test_X, test_Y, batch_size=PARAMS.get('batch_size'))
print('test loss, test mae:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(test_X[:3])
print('predictions shape:', predictions.shape)
model.save('model/model_{}.h5'.format(name))

# finished
