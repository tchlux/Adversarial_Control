import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

import utils

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

# # List of images and steering angles for generating adversarial inputs
# image_and_steering = []
# num_pictures = 1700

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))

            # # Store the image and steering angle
            # image_and_steering.append( (image.copy(), steering_angle) )
            # if len(image_and_steering) > num_pictures:
            #     import pickle
            #     with open("images_and_steering_angles.pkl", "wb") as f:
            #         pickle.dump(image_and_steering, f)
            #     exit()

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


# Read in a set of sample images + correct turning angles.

# Identify the bounds of values that can be placed into the different
# array positions.

# Use generic optimization (AMPGO + L-BFGS-B) to identify minimal
# adversarial perturbations that are robust to a series of of
# random transformations and work in all of the sample images.

# Use Cleverhans to generate adversarial images at each of the
# sample images.



# import argparse
# import base64
# from datetime import datetime
# import os
# import shutil

# import numpy as np
# import socketio
# import eventlet
# import eventlet.wsgi
# from PIL import Image
# from flask import Flask
# from io import BytesIO

# from keras.models import load_model

# import utils

# # Import the reinforcement learning statistics
# from reinforce import current_statistics, new_reinforcement_model, reset
# # Use the training testing splits to validate performance
# from sklearn.model_selection import train_test_split

# sio = socketio.Server()
# app = Flask(__name__)
# model = None
# prev_image_array = None
# steering_angle = 0

# MAX_SPEED = 25
# MIN_SPEED = 10

# speed_limit = MAX_SPEED

# # Holder for (image, steering angle, error)
# image_data = []
# sangle_data = []
# error_data = []
# # Previous steering angle (for recording decisions)
# previous_steering_angle = 0

# TRAINING_BATCH_SIZE = 100
# VALIDATION_PERCENTAGE = 0.1
# RL_MODEL_NAME = "model_frame-%i.h5"
# import time
# last_reset = time.time()
# # Use this generic minimizer in order to find the best steering angle
# from scipy.optimize import minimize

# @sio.on('telemetry')
# def telemetry(sid, data):
#     # These lists will be kept between iterations
#     global image_data, sangle_data, error_data, \
#         previous_steering_angle, last_reset

#     # If data was sent over the web socket (images from car)
#     if data:
#         # The current steering angle of the car
#         steering_angle = float(data["steering_angle"])
#         # The current throttle of the car
#         throttle = float(data["throttle"])
#         # The current speed of the car
#         speed = float(data["speed"])
#         # The current image from the center camera of the car
#         image = Image.open(BytesIO(base64.b64decode(data["image"])))
#         try:
#             image = np.asarray(image)       # from PIL image to numpy array
#             image = utils.preprocess(image) # apply the preprocessing
#             image = np.array(image)       # the model expects 4D array

#             # frame_num   -- iteration of global game execution
#             # dist        -- distance from driving waypoint line
#             # car_speed   -- speed of car in units / frame
#             # dist_change -- change in distance since previous check
#             frame_num, dist, car_speed, dist_change = current_statistics()

#             # Append reward data to current batch
#             image_data.append(image)
#             sangle_data.append(np.array([previous_steering_angle]))
#             error_data.append(np.array([dist]))

#             # If a batch of data has been collected, train the model more
#             if len(image_data) >= TRAINING_BATCH_SIZE:
#                 print("Training a batch...")
#                 train_size = int(len(image_data)*(100-VALIDATION_PERCENTAGE)+0.5)
#                 size = len(image_data)
#                 np_images = np.array(image_data).reshape((size,)+image_data[0].shape)
#                 np_sangle = np.array(sangle_data).reshape((size,)+sangle_data[0].shape)
#                 np_errors = np.array(error_data).reshape((size,)+error_data[0].shape)
#                 i_train, s_train, e_train = (np_images[:train_size],
#                                              np_sangle[:train_size],
#                                              np_errors[:train_size])
#                 i_valid, s_valid, e_valid = (np_images[train_size+1:],
#                                              np_sangle[train_size+1:],
#                                              np_errors[train_size+1:])
#                 # model.fit([i_train, s_train], e_train)
#                 model.train_on_batch([i_train, s_train], e_train)
#                 # Save updated model.
#                 model.save_weights(RL_MODEL_NAME%(frame_num))
#                 # Reset the lists of batch-data
#                 image_data = []
#                 sangle_data = []
#                 error_data = []
#                 print()

#             # Find the steering angle that minimize
