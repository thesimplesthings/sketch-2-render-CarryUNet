import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

#========= Author
'''
Miguel Campillos
miguelcampillos.com
'''
#========= DATA

set_folder = "data_bot"
input_dir = set_folder+r"/input"
target_dir = set_folder+"/target"

img_size = (256,256)


def load_images_from_folder(folder, image_size):
    images = []
    filenames = sorted(os.listdir(folder), key=lambda x: int(os.path.splitext(x)[0]))
    for filename in filenames:
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

X = load_images_from_folder(input_dir, img_size)
Y = load_images_from_folder(target_dir, img_size)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#========= /DATA

#========= CALLBACKS

base_patience = 20

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=base_patience * 2,
)

reduce_learning = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=base_patience)

save_best = keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_loss"
    )


class MonitoringCallback(keras.callbacks.Callback):
    def __init__(self,
                 sample_input: np.ndarray,
                 sample_target: np.ndarray,
                 out_dir: str = "monitor",
                 n_epochs: int = 1):
        super().__init__()
        self.sample_input  = sample_input.astype(np.float32)  
        self.sample_target = sample_target.astype(np.float32) 
        self.out_dir       = out_dir
        self.n_epochs      = n_epochs
        os.makedirs(self.out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # only run every n_epochs
        if (epoch + 1) % self.n_epochs != 0:
            return

        # predict
        x = np.expand_dims(self.sample_input, axis=0) 
        y_pred = self.model.predict(x)[0]              
        
        y_pred = y_pred.astype(np.float32)

        # plot side-by-side: input | target | pred
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes,
                                  [self.sample_input,
                                   self.sample_target,
                                   y_pred],
                                  ["Input", "Target", f"Pred @ epoch {epoch+1}"]):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis("off")

        # save
        fname = os.path.join(self.out_dir, f"monitor_epoch_{epoch+1:03d}.png")
        fig.savefig(fname, bbox_inches="tight")
        
        plt.show()
        plt.close(fig)
        
sample_idx = 2
sample_x = X_train[sample_idx]
sample_y = y_train[sample_idx]

monitor_cb = MonitoringCallback(
    sample_input=sample_x,
    sample_target=sample_y,
    out_dir="training_monitor",
    n_epochs=5   # every n epochs
)

#========= /CALLBAKS

#========= CUSTOM LAYERS AND ACTS

class InvertLayer(Layer):
    def call(self, inputs):
        return 1.0 - inputs
    

def make_mask(y_true, thresh=0.9):
   
    gray = tf.reduce_max(y_true, axis=-1, keepdims=True)

    mask = tf.cast(gray < thresh, tf.float32)
    
    mask = tf.nn.avg_pool2d(mask, ksize=7, strides=8, padding="SAME")
    mask = tf.image.resize(mask,
                            size=[256, 256],
                            method=tf.image.ResizeMethod.GAUSSIAN)
    return mask

#plt.imshow(make_mask(y_train[2:2+1])[0, ..., 0], cmap='gray')

d = 6.0          # up-weight foreground; tune 4–10
eps = 1e-6       # avoid division by zero

def masked_L1(y_true, y_pred):
    mask = make_mask(y_true)

    fg_err = tf.abs(y_true - y_pred) * mask
    bg_err = tf.abs(y_true - y_pred) * (1.0 - mask)


    fg_norm = tf.maximum(tf.reduce_mean(mask), eps)
    bg_norm = tf.maximum(tf.reduce_mean(1.0 - mask), eps)

    loss = d * tf.reduce_mean(fg_err) / fg_norm + tf.reduce_mean(bg_err) / bg_norm
    return loss

#========= CUSTOM LAYERS AND ACTS

#========= MODEL

act = "leaky_relu"
act2 = "relu"

scale = 1

def build_pix2pix(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Dist
    # x = layers.RandomRotation(5/360)(inputs)

    # === Encoder ===
    e1 = layers.SeparableConv2D(int(64*scale), 4, strides=2, padding="same", activation=act2)(inputs)   # 128x128
    
    e2 = layers.Conv2D(int(128*scale), 4, strides=2, padding="same", activation=act2)(e1) # 64x64
    e3 = layers.Conv2D(int(256*scale), 4, strides=2, padding="same", activation=act2)(e2) # 32x32
    e4 = layers.Conv2D(int(512*scale), 4, strides=2, padding="same", activation=act2)(e3) # 16x16

    # === Bottleneck ===
    b = layers.Conv2D(int(512*scale), 4, strides=2, padding="same", activation=act2)(e4)  # 8x8

    # === Decoder ===
    d1 = layers.Conv2DTranspose(int(512*scale), 4, strides=2, padding="same", activation=act)(b)  # 16x16
    
    #instead encoder-> decoder skip, I prepare decoder -> decoder. I want to expand the low level information to reinforce later layers
    d10 = layers.AveragePooling2D(pool_size=(3,3), strides=1,padding="same", name="box_blur2")(d1)
    d10 = layers.UpSampling2D(size=(8,8),interpolation="nearest")(d10)
    
    d1 = layers.Dropout(0.3)(d1)

    d2 = layers.Conv2DTranspose(int(128*scale), 4, strides=2, padding="same", activation=act2,use_bias=True)(d1)  # 32x32
    d2 = layers.Concatenate()([d2, e3])

    d3 = layers.Conv2DTranspose(int(64*scale), 4, strides=2, padding="same", activation=act2,use_bias=True)(d2)  # 64x64
    d3 = layers.Concatenate()([d3, e2])
    
    #recovering low level information here
    d4 = layers.Conv2DTranspose(int(32*scale), 4, strides=2, padding="same", activation=act,use_bias=False)(d3)   # 128x128
    d4 = layers.Concatenate()([d4,d10])
    
    d5 = layers.UpSampling2D(size =(2,2),interpolation="bilinear")(d4)

    outputs = layers.Conv2D(3, kernel_size=3, padding="same", activation="sigmoid", use_bias=True)(d5)

    model = models.Model(inputs, outputs, name="pixtest")
    return model

model = build_pix2pix()

#model.load_weights("best_model.keras")

opt ="adam"
model.compile(optimizer=opt, loss="mse", metrics=["mse"])
model.summary()

#========= /MODEL

#========= TRAINING

history = model.fit(
    x = X_train,
    y = y_train,
    batch_size=8, 
    epochs=200, 
    callbacks=[save_best, reduce_learning, early_stop,monitor_cb],
    validation_data=(X_test,y_test)
    )

model.save("EndOfTrainingModel.keras")

try:
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
except NameError:
    print("Error: `history` not found. Ensure you have the training `history` object from `model.fit` in scope.")
else:
    epochs = range(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, color='red', label='Train Loss')
    plt.plot(epochs, val_loss, color='blue', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()