# imports
import neptune
import neptune_tensorboard as neptune_tb
import numpy as np
# set project and start integration with keras
from keras import Input, Model, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Activation, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
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
    'num_cnn_blocks': 3,
    'num_filters': 64,
    'kernel_size': 4,
    'num_dense_nodes': 1024,
    'dense_nodes_divisor': 4,
    'batch_size': 64,
    'drop_out': 0.1,
    'learning_rate': 0.001,
    'input_shape': (VECTOR_SIZE, 1, 1)
}

# start experiment

name = 'keras-integration-cnn'

if LOG_NEPTUNE:
    neptune.init(project_qualified_name='4ND4/sandbox')
    neptune_tb.integrate_with_keras()
    result = neptune.create_experiment(name=name, params=PARAMS)

    name = result.id

# start of cnn coding
input_tensor = Input(shape=PARAMS.get('input_shape'))

# 1st cnn block
x = BatchNormalization()(input_tensor)
x = Activation('relu')(x)
x = Conv2D(filters=PARAMS['num_filters'],
           kernel_size=PARAMS['kernel_size'],
           strides=1, padding='same')(x)
# x = MaxPooling2D()(x)
x = Dropout(PARAMS['drop_out'])(x)

# additional cnn blocks
for iblock in range(PARAMS['num_cnn_blocks'] - 1):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=PARAMS['num_filters'],
               kernel_size=PARAMS['kernel_size'],
               strides=1, padding='same')(x)
    x = MaxPooling2D(padding='same')(x)
    x = Dropout(PARAMS['drop_out'])(x)

# mlp
x = Flatten()(x)
x = Dense(PARAMS['num_dense_nodes'], activation='relu')(x)
x = Dropout(PARAMS['drop_out'])(x)
x = Dense(PARAMS['num_dense_nodes'] // PARAMS['dense_nodes_divisor'],
          activation='relu')(x)
output_tensor = Dense(1, activation='linear')(x)

# instantiate and compile model
model = Model(inputs=input_tensor, outputs=output_tensor)

optimizer = Adam(lr=PARAMS.get('learning_rate'))  # default = 0.001

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])


def getdata():
    _data = np.load('output/DeepUAge-faces-embeddings_{}_{}.npz'.format(VECTOR_SIZE, FACE_DETECTION))
    embeddings_val_npz_path = 'output/DeepUAge-val-faces-embeddings-{}-{}.npz'.format(VECTOR_SIZE, FACE_DETECTION)

    trainX, trainy, valX, valy = _data['arr_0'], _data['arr_1'], _data['arr_2'], _data['arr_3']
    # print('Dataset: train=%d, test=%d' % (trainX.shape[0], valX.shape[0]))

    # test data

    if DEBUG:
        resize_value = 100

        trainX, trainy, valX, valy = trainX[0:resize_value], trainy[0:resize_value], valX[0:resize_value], valy[
                                                                                                           0:resize_value]
        print('Dataset resized: train=%d, test=%d' % (trainX.shape[0], valX.shape[0]))

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


train_X, train_Y, val_X, val_Y, test_X, test_Y = getdata()

img_rows = 512
img_cols = 1

train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)
val_X = val_X.reshape(val_X.shape[0], img_rows, img_cols, 1)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

checkpoint = ModelCheckpoint('checkpoint/model-{epoch:03d}-{val_loss:03f}-{val_mae:03f}.h5', save_best_only=True,
                             monitor='val_loss', mode='min')

model.fit(
    train_X, train_Y,
    validation_data=(val_X, val_Y),
    epochs=PARAMS['epoch_nr'],
    batch_size=PARAMS['batch_size'],
    callbacks=[es, checkpoint]
)

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
