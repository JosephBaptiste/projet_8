import sys
from traceback import print_tb
import pickle

import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import numpy as np

def rle_decode_tf(mask_rle, shape):
    # Copied from https://stackoverflow.com/questions/58693261/decoding-rle-run-length-encoding-mask-with-tensorflow-datasets
    shape = tf.convert_to_tensor(shape, tf.int64)
    size = tf.math.reduce_prod(shape)
    # Split string
    s = tf.strings.split(mask_rle)
    s = tf.strings.to_number(s, tf.int64)
    # Get starts and lengths
    starts = s[::2] - 1
    lens = s[1::2]
    # Make ones to be scattered
    total_ones = tf.reduce_sum(lens)
    ones = tf.ones([total_ones], tf.uint8)
    # Make scattering indices
    r = tf.range(total_ones)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
    # Scatter ones into flattened mask
    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    # Change shape to add channel for resize (to delete if not working)
    shape = (1, shape[0], shape[1])
    # Reshape into mask
    return tf.reshape(mask_flat, shape)

def resize(new_input):
    new_input = tf.image.resize(new_input, (1024, 1024), method="nearest")
    return new_input

def get_df():
    datadir = "./data/"
    # lire le csv d'entraînement
    df = pd.read_csv(datadir+"train.csv")
    # ajouter le path de chaque image
    df['img_path'] = df['id'].apply(lambda x: f"{datadir}/train_images/{x}.tiff")
    # suppression des colonnes non utilisées
    df.drop(['data_source', 'pixel_size', 'tissue_thickness', 'age','sex'], axis=1, inplace=True)
    return df

def decode_tiff_experimental(img_path):
    img = tf.io.read_file(img_path)
    img = tfio.experimental.image.decode_tiff(img)
    return img

def preprocessing_crop2(img_path: str, mask_rle: str, target_height:int = 3000, target_width:int = 3000): 
    """
    Huge thanks to hoyso48 for the help! (https://www.kaggle.com/competitions/hubmap-organ-segmentation/discussion/347568)
    LorSong as well (https://github.com/tensorflow/tensorflow/issues/6743)
    
    """
    # Decode entire image
    img = decode_tiff_experimental(img_path) 
    # increase contrast
    #img = tf.image.adjust_contrast(img, 2.)
    # Set data between 0 and 1
    img = tf.cast(img,  tf.float32) / 255.
    # removing alpha channel
    img = img[:,:,:3]
    # resizing
    img = resize(img)
    # grayscaling, 
    img = tf.image.rgb_to_grayscale(img)

    # Decode entire mask
    mask = tf.transpose(rle_decode_tf(mask_rle, (target_width, target_height)))
    # resize mask
    mask = resize(mask)

    # cropping image:
    img = tf.expand_dims(img, axis=0)
    img = tf.image.extract_patches(images=img, 
                                  sizes=[1, 256, 256, 1],
                                  strides=[1, 256, 256 ,1],
                                  rates=[1, 1, 1, 1],
                                  padding='SAME')
    # reshaping into image
    img = tf.reshape(img, (-1, 256,256))
    # cropping mask:
    mask = tf.expand_dims(mask, axis=0)
    mask = tf.image.extract_patches(images=mask, 
                                  sizes=[1, 256, 256, 1],
                                  strides=[1, 256, 256 ,1],
                                  rates=[1, 1, 1, 1],
                                  padding='VALID')
    # reshaping into image
    mask = tf.reshape(mask, (-1, 256,256))
    return img, mask

def get_dataset(df: pd.DataFrame) -> Dataset:
    """
    """
    # Split data between test and train sets
    train_df, test_df = train_test_split(df, train_size=0.7, stratify=df['organ'])
    # Create dataset, then apply preprocessing function
    ds_train = Dataset.from_tensor_slices((train_df['img_path'].values, 
                                           train_df['rle'].values, 
                                           train_df['img_height'].values,
                                           train_df['img_width'].values))
    ds_train = ds_train.map(preprocessing_crop2, tf.data.experimental.AUTOTUNE)
    # same for testing
    ds_test = Dataset.from_tensor_slices((test_df['img_path'].values, 
                                          test_df['rle'].values, 
                                          test_df['img_height'].values, 
                                          test_df['img_width'].values))
    ds_test = ds_test.map(preprocessing_crop2, tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test



def get_unet_model():
    # From https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
    def double_conv_block(x, n_filters):
        # Conv2D then ReLU activation
        #x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        # Conv2D then ReLU activation
        #x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

        return x

    def downsample_block(x, n_filters):
        f = double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p

    def upsample_block(x, conv_features, n_filters):
        # upsample
        #x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = layers.Conv2DTranspose(n_filters, 1, 2, padding="same")(x)
        # concatenate
        x = layers.concatenate([x, conv_features])
        # dropout
        x = layers.Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = double_conv_block(x, n_filters)
        return x
    
    # inputs
    #inputs = layers.Input(shape=(128,128,3))
    inputs = layers.Input(shape=(256,256,1))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    #outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model

def get_data_for_predict() -> pd.DataFrame:
    datadir = "./data/"
    df_for_predict = pd.read_csv(datadir+"test.csv")
    df_for_predict['img_path'] = df_for_predict['id'].apply(lambda x: f"{datadir}/test_images/{x}.tiff")
    return df_for_predict

def simple_rgb_to_grayscale(image: np.array):
    # Same weights used by tensorflow for tf.image.rgb_to_grayscale
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray_image = image.numpy().dot(rgb_weights)
    return tf.convert_to_tensor(gray_image)

def get_values_for_crop(img_total_height: int = 1024, img_total_width: int = 1024, output_size: int = 256) -> list:
    """
    with given height and width, return list of offset values for tf.image.crop_to_bounding_box
    by default, no overlap, meaning that two adjacent borders of width: total % output_size will be cropped out.
    with overlap, boundaries are moved of only half the value (approximately doubling the length of returned list).
    """
    # Creating lists:
    list_height_offset = [x for x in range(0, img_total_height - output_size + 1, output_size)]
    list_width_offset = [x for x in range(0, img_total_width - output_size + 1, output_size)]
    dict_hw = dict(zip(list_height_offset, [[] for x in range(len(list_height_offset))]))
    for h in list_height_offset:
        for w in list_width_offset:
            dict_hw[h].append(w)
    return dict_hw

def rle_encode(mask: np.array) -> str:
    """ 
    mask: numpy array, 1 - mask, 0 - background
    returns run-length as string formated
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_submit_csv(data_for_pred: pd.DataFrame, model):
    result_df = pd.DataFrame(columns=['id', 'rle'])
    datadir = "./data/"
    shape_val = 256
    for idx, line in data_for_pred.iterrows():
        img = decode_tiff_experimental(f"{datadir}test_images/{line['id']}.tiff") 
        # removing alpha channel
        img = img[:,:,:3]
        # resizing
        img = resize(img)
        # grayscaling, 
        img = simple_rgb_to_grayscale(img)
        img = tf.cast(img,  tf.float32) / 255.
        mask_pred = None
        val_temp = None
        for height, list_width in get_values_for_crop().items():
            array_temp = None
            for width in list_width:
                img_crop_for_pred = img[height:height+shape_val,
                                                        width:width+shape_val]
                pred_from_crop =  model.predict(np.expand_dims(img_crop_for_pred, axis=0))
                if array_temp is None:
                    array_temp = pred_from_crop[0]
                else:
                    array_temp = np.hstack([array_temp, pred_from_crop[0]])
            if mask_pred is None:
                mask_pred = array_temp
            else:
                mask_pred = np.vstack([mask_pred, array_temp])
        val = 0.18
        mask_pred[mask_pred < val] = 0
        mask_pred[mask_pred > 0] = 1
        result_df = result_df.append(pd.Series([line['id'], rle_encode(mask_pred)], index=['id', 'rle']), ignore_index=True)
    result_df.to_csv('./submission.csv', index=False)
    return result_df

def Main():
    # Récupérer les données
    df = get_df()
    ds_train, ds_test = get_dataset(df)
    BATCH_SIZE = 32
    BUFFER_SIZE = 1024
    train_batches = ds_train.unbatch().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_batches = ds_test.unbatch().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()
    test_batches = test_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # récupérer et entraîner le modèle
    unet_model = get_unet_model()
    unet_model.compile(
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.009, momentum=0.9),
                        loss=keras.losses.BinaryCrossentropy(),
                        metrics= keras.metrics.BinaryIoU()
                    )
    try:
        model_history = unet_model.fit(train_batches,
                                        validation_data = test_batches,
                                        epochs=100,
                                        steps_per_epoch=10,  
                                        validation_steps=10,
                                        )
    except Exception as e:
        print(e)
        t, v, tb = sys.exc_info()
        print(t, v)
        print_tb(tb)
        print("Problème rencontré lors de l'entrainement du modèle; le programme s'arrête.")
        sys.exit()
    try:
        unet_model.save('./unet_model.h5')
        with open('./history.pkl', "wb") as file:
            pickle.dump(model_history, file)
    except Exception as e:
        print(e)
        t, v, tb = sys.exc_info()
        print(t, v)
        print_tb(tb)
        print("Problème rencontré lors de la sauvegarde du modèle; le programme continue.")
    print("Entraînement terminé, prédiction depuis test.csv:")
    df_pred = get_data_for_predict()
    get_submit_csv(df_pred, unet_model)

if __name__ == "__main__":
    Main()