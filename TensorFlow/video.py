
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf


""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("videos/outputs")

    """ Loading Model """
    model = tf.keras.models.load_model("files/model.h5")
    # model.summary()

    """ Video path """
    video_name = "video-1"
    video_path = f"videos/inputs/{video_name}.mp4"

    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    h, w, _ = frame.shape
    vs.release()

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(f'videos/outputs/{video_name}.avi', fourcc, 30, (w, h), True)

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            out.release()
            break

        h, w, _ = frame.shape
        ori_frame = frame
        frame = cv2.resize(frame, (W, H)) ## (H, W, 3)
        frame = np.expand_dims(frame, axis=0) ## (1, H, W, 3)
        frame = frame / 255.0

        mask = model.predict(frame, verbose=0)[0][:,:,-1]
        mask = cv2.resize(mask, (w, h)) ## (h, w)
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)    ## (h, w, 1)

        photo_mask = mask
        background_mask = np.abs(1-mask) ## (h, w, 1)
        masked_frame = ori_frame * photo_mask

        background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1) ## (h, w, 3)
        background_mask = background_mask * [255, 0, 255]
        final_frame = masked_frame + background_mask
        final_frame = final_frame.astype(np.uint8)

        out.write(final_frame)

    print("Completed!")




    ##
