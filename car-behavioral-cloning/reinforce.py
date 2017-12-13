import os, time
from util.system import run
import numpy as np

def get_xyz():
    x,y,z,frame = list(map(float,run(["cat","/home/thomas/ML_Research/self-driving-car-sim/Assets/1_SelfDrivingCar/Text/carlocation.txt"])[1][0].split()))
    return np.array([x,y,z]),frame

def check_dist(position):
    return float(run(["./wpdist"]+list(map(str,list(position))))[1][0].strip())

def reset():
    with open("/home/thomas/ML_Research/self-driving-car-sim/Assets/1_SelfDrivingCar/Text/reset.txt", "w") as f:
        f.write("reset")
    time.sleep(2)

POSITION,FRAME_NUM = get_xyz()
DIST = check_dist(POSITION)

# Function for returning the current statistics of the car. This function
def current_statistics():
    # Make sure the globals are used for storing these persistent values
    global DIST, FRAME_NUM, POSITION
    # Transition globals to old values
    prev_dist = DIST
    prev_frame_num = FRAME_NUM
    prev_position = POSITION
    speed = None
    dist_change = -1
    try:
        POSITION, FRAME_NUM = get_xyz()
        DIST = check_dist(POSITION)
        dist_change = DIST - prev_dist
        if (prev_frame_num != FRAME_NUM):
            speed = np.sqrt(np.sum((prev_position-POSITION)**2)) / (FRAME_NUM-prev_frame_num)
    except:
        # Occasionally there are file I/O errors from the system,
        # ignore those rounds entirely
        pass
    return FRAME_NUM, DIST, speed, dist_change
    

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Merge
from utils import INPUT_SHAPE

DROPOUT_KEEP_PROBABILITY = 0.5

# Function for creating a reinforcement learning model. Similar to a
# proximal policy optimization model, learns the gradient of the
# reward with respect to changes in decisions. Allows for the
# identification of an optimal decision using gradient descent / ascent.
def new_reinforcement_model():
    """
    Modified NVIDIA model for reinforcement learning. Takes inputs
    [image, steering angle] numpy arrays of "INPUT_SHAPE" and "1" respectively.
    """

    # Generate the convolutional model for processing image input
    conv_model = Sequential()
    conv_model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    conv_model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    conv_model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    conv_model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    conv_model.add(Conv2D(64, 3, 3, activation='elu'))
    conv_model.add(Conv2D(64, 3, 3, activation='elu'))
    conv_model.add(Dropout(DROPOUT_KEEP_PROBABILITY))
    conv_model.add(Flatten())
    
    # Generate a model that takes a single number as input for steering
    steering_angle_model = Sequential()
    steering_angle_model.add(Dense(1, activation='elu', input_shape=(1,)))

    # Generate a model that is the composition of the image and
    # steering angle
    rl_model = Sequential()
    rl_model.add(
        Merge([conv_model, steering_angle_model], mode="concat")
    )
    # Do standard deep-learning on the combined image features +
    # steering angle outputs.
    rl_model.add(Dense(100, activation='elu'))
    rl_model.add(Dense(50, activation='elu'))
    rl_model.add(Dense(10, activation='elu'))
    rl_model.add(Dense(1))
    # Print out a summary of the model that was created
    rl_model.summary()

    # Compile the model so that it is ready to be trained. Use the
    # Adam algorithm for training with a learning rate of 10^(-4)
    rl_model.compile(loss='mean_squared_error', optimizer=Adam(lr=10**(-4)))

    return rl_model



if __name__ == "__main__":
    while True:
        frame_num, dist, speed, dist_change = current_statistics()
        if speed > 0:
            print("","%i: %.2f  %.2f  %.2f"%(frame_num, dist, speed, dist_change))


# Train a model (of the same shape + inputs like 'current reward') to
# estimate the reward for a decision
