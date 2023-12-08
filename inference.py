from other_functions import *



# set paths:
ship_dir = '../input/airbus-ship-detection'
test_image_dir = os.path.join(ship_dir, 'test_v2') # way to dataset on kaggle
test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')


# load the best model: 

fullres_model = models.load_model('/model/model_all_2_0.4181.h5', compile=False)
seg_in_shape = fullres_model.get_input_shape_at(0)[1:3]
seg_out_shape = fullres_model.get_output_shape_at(0)[1:3]
print(seg_in_shape, '->', seg_out_shape)


# see predictions:
fig, m_axs = plt.subplots(8, 2, figsize = (10, 40))
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    first_img = np.expand_dims(c_img, 0)/255.0
    first_seg = fullres_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')


# # other way to see predictions:

# from tqdm import tqdm_notebook
# from skimage.morphology import binary_opening, disk

# out_pred_rows = []
# for c_img_name in tqdm_notebook(test_paths):
#     c_path = os.path.join(test_image_dir, c_img_name)
#     c_img = imread(c_path)
#     c_img = np.expand_dims(c_img, 0)/255.0
#     cur_seg = fullres_model.predict(c_img)[0]
#     cur_seg = binary_opening(cur_seg>0.5, np.expand_dims(disk(2), -1))
#     cur_rles = multi_rle_encode(cur_seg)
#     if len(cur_rles)>0:
#         for c_rle in cur_rles:
#             out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': c_rle}]
#     else:
#         out_pred_rows += [{'ImageId': c_img_name, 'EncodedPixels': None}]
#     gc.collect()