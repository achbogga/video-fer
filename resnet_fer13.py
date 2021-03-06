#!/usr/bin/env python
from keras resnet import *
import os
import sys
import argparse

# Model parameter
	# ----------------------------------------------------------------------------
	#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
	# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
	#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
	# ----------------------------------------------------------------------------
	# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
	# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
	# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
	# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
	# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
	# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
	# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
	# ---------------------------------------------------------------------------

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet_depth_n', type=int, help='The depth of the ResNet network. For ex: default depth is 3. i.e., for v1, it is 3*6 + 2 = 20; for v2, 3*9 + 2 = 29', default = 3)
    parser.add_argument('--resnet_version', type=int, help='The version of the ResNet network. Default is v1.', default = 1)
    parser.add_argument('--nb_epochs', type=int, help='The number of epochs to be trained', default = 200)
    parser.add_argument('--augment_data', type=int, help='Flag whether to augment data or not', default = 1)
    parser.add_argument('--batch_size', type=int, help='batch_size', default = 128)
    parser.add_argument('--subtract_pixel_mean', type=int, help='Flag whether to subtract_pixel_mean from the training data or not', default = 1)

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    # Training parameters
	batch_size = 128  # orig paper trained all networks with batch_size=128
	epochs = 200
	data_augmentation = True
	num_classes = 7
	# Subtracting pixel mean improves accuracy
	subtract_pixel_mean = True
	n = 3
	# Model version
	# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
	version = 1
	# Computed depth from supplied model parameter n
	if version == 1:
	    depth = n * 6 + 2
	elif version == 2:
	    depth = n * 9 + 2
	# Model name, depth and version
	model_type = 'ResNet%dv%d' % (depth, version)

	# Load the CIFAR10 data.
	#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	x_train = np.load('/home/ovuser/Projects/video-fer/Data/FER2013/X_train_48_1.npy')
	x_test = np.load('/home/ovuser/Projects/video-fer/Data/FER2013/X_test_48_1.npy')
	y_train = np.load('/home/ovuser/Projects/video-fer/Data/FER2013/Y_train_48_1.npy')
	y_test = np.load('/home/ovuser/Projects/video-fer/Data/FER2013/Y_test_48_1.npy')

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
	    x_train_mean = np.mean(x_train, axis=0)
	    x_train -= x_train_mean
	    x_test -= x_train_mean

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print('y_train shape:', y_train.shape)

	# Convert class vectors to binary class matrices.
	#y_train = keras.utils.to_categorical(y_train, num_classes)
	#y_test = keras.utils.to_categorical(y_test, num_classes)

	if version == 2:
	    model = resnet_v2(input_shape=input_shape, depth=depth, num_classes = 7)
	else:
	    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes = 7)

	model.compile(loss='categorical_crossentropy',
	              optimizer=Adam(lr=lr_schedule(0)),
	              metrics=['accuracy'])
	model.summary()
	print(model_type)
	# Prepare model model saving directory.
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'fer2013_%s_model.{epoch:03d}.h5' % model_type
	if not os.path.isdir(save_dir):
	    os.makedirs(save_dir)
	filepath = os.path.join(save_dir, model_name)

	# Prepare callbacks for model saving and for learning rate adjustment.
	checkpoint = ModelCheckpoint(filepath=filepath,
	                             monitor='val_acc',
	                             verbose=1,
	                             save_best_only=True)

	lr_scheduler = LearningRateScheduler(lr_schedule)

	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
	                               cooldown=0,
	                               patience=5,
	                               min_lr=0.5e-6)

	callbacks = [checkpoint, lr_reducer, lr_scheduler]

	# Run training, with or without data augmentation.
	if not data_augmentation:
	    print('Not using data augmentation.')
	    model.fit(x_train, y_train,
	              batch_size=batch_size,
	              epochs=epochs,
	              validation_data=(x_test, y_test),
	              shuffle=True,
	              callbacks=callbacks)
	else:
	    print('Using real-time data augmentation.')
	    # This will do preprocessing and realtime data augmentation:
	    datagen = ImageDataGenerator(
	        # set input mean to 0 over the dataset
	        featurewise_center=False,
	        # set each sample mean to 0
	        samplewise_center=False,
	        # divide inputs by std of dataset
	        featurewise_std_normalization=False,
	        # divide each input by its std
	        samplewise_std_normalization=False,
	        # apply ZCA whitening
	        zca_whitening=False,
	        # epsilon for ZCA whitening
	        zca_epsilon=1e-06,
	        # randomly rotate images in the range (deg 0 to 180)
	        rotation_range=0,
	        # randomly shift images horizontally
	        width_shift_range=0.1,
	        # randomly shift images vertically
	        height_shift_range=0.1,
	        # set range for random shear
	        shear_range=0.,
	        # set range for random zoom
	        zoom_range=0.,
	        # set range for random channel shifts
	        channel_shift_range=0.,
	        # set mode for filling points outside the input boundaries
	        fill_mode='nearest',
	        # value used for fill_mode = "constant"
	        cval=0.,
	        # randomly flip images
	        horizontal_flip=True,
	        # randomly flip images
	        vertical_flip=False,
	        # set rescaling factor (applied before any other transformation)
	        rescale=None,
	        # set function that will be applied on each input
	        preprocessing_function=None,
	        # image data format, either "channels_first" or "channels_last"
	        data_format=None,
	        # fraction of images reserved for validation (strictly between 0 and 1)
	        validation_split=0.0)

	    # Compute quantities required for featurewise normalization
	    # (std, mean, and principal components if ZCA whitening is applied).
	    datagen.fit(x_train)

	    # Fit the model on the batches generated by datagen.flow().
	    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
	                        validation_data=(x_test, y_test),
	                        epochs=epochs, verbose=1, workers=4,
	                        callbacks=callbacks)

	# Score trained model.
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
