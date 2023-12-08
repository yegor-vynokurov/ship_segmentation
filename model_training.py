from other_functions import *

BATCH_SIZE = 4
EDGE_CROP = 16
NB_EPOCHS = 50
GAUSSIAN_NOISE = 0.1
# UPSAMPLE_MODE = 'DECONV'
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 500
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False




# Read a dataframe with masks and image's ids. 
masks = pd.read_csv(os.path.join('../input/airbus-ship-detection',
                                 'train_ship_segmentations_v2.csv'))
# load unique and good quality images
unique_img_ids= pd.read_pickle ("/support/unique_img_ids.pkl")

# divide into train and validation dataset
from sklearn.model_selection import train_test_split
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.3, shuffle = True,
                 stratify = unique_img_ids['ships'], random_state = 400) # let's stratify by goal and unbalanced column.
train_df = pd.merge(masks, train_ids) # add masks to train dataset
valid_df = pd.merge(masks, valid_ids) # add masks to validation dataset 

# choose only images with ships
train_df = train_df[train_df['ships'] != 0]
valid_df = valid_df[valid_df['ships'] != 0]
train_df.shape, valid_df.shape

train_gen = make_image_gen(train_df, 4) # create a generator
valid_gen = make_image_gen(valid_df, VALID_IMG_COUNT)

# set an augmentations

dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
# brightness can be problematic since it seems to change the labels differently from the images 
if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)




# take an augmentation
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)


# Build U-Net model
if UPSAMPLE_MODE=='DECONV':
    upsample=upsample_conv
else:
    upsample=upsample_simple
    
input_img = layers.Input(t_x.shape[1:], name = 'RGB_Input')
pp_in_layer = input_img
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer) # for built-in scaling of images
    
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer) # for simplificate of images
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model_all = models.Model(inputs=[input_img], outputs=[d])
seg_model_all.summary()


# compile of model
seg_model_all.compile(optimizer=Adam(1e-4), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

valid_x, valid_y = next(valid_gen)


# choose path to load of model
path = f'/model' # 


# add a callbacks
checkpoint = ModelCheckpoint(filepath=path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = False, save_format="tf")

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(train_df))


# fit the model
loss_history = [seg_model_all.fit(aug_gen, 
                                  steps_per_epoch=step_count, 
                                  epochs=NB_EPOCHS, 
                                  validation_data=(valid_x, valid_y.astype(np.float32)),
                                  callbacks=callbacks_list,
                                  workers=1 # the generator is not very thread safe
                                       )]


show_loss(loss_history)

# seg_model.load_weights(weight_path)
# seg_model_all.save('seg_model_2.h5')


# see an example:
valid_x, valid_y = next(valid_gen)

pred_y = seg_model_all.predict(valid_x)
print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.hist(pred_y.ravel(), np.linspace(0, 1, 10))
ax.set_xlim(0, 1)
ax.set_yscale('log')
