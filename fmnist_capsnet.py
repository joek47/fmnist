import argparse
import os
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from keras import layers, models, optimizers
from keras import backend as K
from keras import callbacks
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.model_selection import train_test_split
from glob import glob

parser = argparse.ArgumentParser(prog="Fashion MNIST Capsule Network")
parser.add_argument('--batch_size', default=128, type=int, help="Default 128")
parser.add_argument('--epochs', default=100, type=int, help="Default 100")
parser.add_argument('--save_dir', default="result", help="Directory for saved weights")
parser.add_argument('--mode', choices=['train', 'test', 'find'])
parser.add_argument('--weights', default=None, help="Load saved weights")
parser.add_argument('--lr', default=0.001, type=float, help="Default 0.001")
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

def create_callbacks(save_dir,batch_size, histogram, lr, monitor='val_capsnet_acc'):
    filepath= save_dir + "/weights-improvement-{epoch:02d}-{val_capsnet_acc:.2f}.hdf5"
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True, mode='max')
    log = callbacks.CSVLogger(save_dir + "/log.csv")
    tb = callbacks.TensorBoard(log_dir=save_dir+ "/tb", batch_size=batch_size, histogram_freq=histogram)
    lr_decay = callbacks.LearningRateScheduler(schedule = lambda epoch: lr * (0.95**epoch))

    return [log, tb, checkpoint, lr_decay]

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
                0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def perform_prediction(emodel):
    y_pred, x_recon = emodel.predict(X_test, batch_size=args.batch_size)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])


data_train = pd.read_csv('input/fashion-mnist_train.csv')
data_test = pd.read_csv('input/fashion-mnist_test.csv')

num_classes = 10

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

#Here we split validation data to optimize classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data
X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([iaa.Fliplr(0.5)])
X_train = seq.augment_images(X_train)

K.set_image_data_format('channels_last')

x = layers.Input(shape=input_shape)

# Layer 1: Just a conventional Conv2D layer
conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

# Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

# Layer 3: Capsule layer. Routing algorithm works here.
digitcaps = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, num_routing=3,
                                 name='digitcaps')(primarycaps)

# Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
# If using tensorflow, this will not be necessary. :)
out_caps = Length(name='capsnet')(digitcaps)

# Decoder network.
y = layers.Input(shape=(num_classes,))
masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

# Shared Decoder model in training and prediction
decoder = models.Sequential(name='decoder')
decoder.add(layers.Dense(512, activation='relu', input_dim=16*num_classes))
decoder.add(layers.Dropout(0.2))
decoder.add(layers.Dense(1024, activation='relu'))
decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

# Models for training and evaluation (prediction)
model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
eval_model = models.Model(x, [out_caps, decoder(masked)])
model.summary()

# compile the model
model.compile(optimizer=optimizers.Adam(lr=args.lr),
              loss=[margin_loss, 'mse'],
                loss_weights=[1., 0.392],
              metrics={'capsnet': 'accuracy'})


if args.weights is not None:
    model.load_weights(args.weights)
    print("loading weights", args.weights)

# Train
if args.mode=='train':
    history = model.fit([X_train, y_train], [y_train, X_train], batch_size=args.batch_size, epochs=args.epochs,
                  verbose=1, validation_data=[[X_val, y_val], [y_val, X_val]], 
                  callbacks=create_callbacks(args.save_dir,  args.batch_size, 1, args.lr, 'val_capsnet_acc'))

elif args.mode=='find':
    g = glob(args.save_dir + '/*.hdf5')

    for wt in g:
        eval_model.load_weights(wt)
        print(wt)
        perform_prediction(eval_model)

elif args.mode=='test':
    perform_prediction(eval_model)

