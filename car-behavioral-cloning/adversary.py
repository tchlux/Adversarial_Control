import pickle
import numpy as np
from PIL import Image
from image import random_transformation
from keras.models import load_model

#      Load the images and correct steering angles     
# =====================================================
data_file = "images_and_steering_angles.pkl"
with open(data_file, "rb") as f:
    images_and_steering = pickle.load(f)

# Load the keras model for driving     
model = load_model("final_model.h5")

# Global variables
IMAGE_SHAPE = images_and_steering[0][0].shape[1:]
RANDOM_SEED = 0
ADVERSARIAL_IMAGE_SIZE = tuple(np.array(IMAGE_SHAPE[:-1]) // 3)+(IMAGE_SHAPE[-1],)
ADVERSARIAL_IMAGE_SCALE = [-1,3]
NEUTRAL_VALUE = 125
NUM_TRANSFORMATIONS = 100
NUM_IMAGES = 50
MAX_COMPUTE_TIME_SEC = ((60*60)*24)

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Image shape:    ", IMAGE_SHAPE)
print("Adversary shape:", ADVERSARIAL_IMAGE_SIZE)

# Given an addition as a Pillow image, generate a randomly transformed
# version of that image as a numpy array.
def generate_transformed_addition(addition):
    transformed_addition = random_transformation(
        addition, s_range=ADVERSARIAL_IMAGE_SCALE)
    # transformed_addition.save("transformed_addition.png")
    transformed_addition = np.array(transformed_addition)
    # Set all alpha=0 pixels to the neutral value (do not use alpha)
    to_delete = (transformed_addition[:,:,-1] == 0)
    transformed_addition[to_delete,:-1] = NEUTRAL_VALUE
    # Transfer the final image to one with the correct shape (in case
    # rounding causes a slight mismatch of shapes) and remove alpha
    transformed_addition = transformed_addition[:,:,:-1]
    output = np.ones(IMAGE_SHAPE, dtype=np.uint8) * NEUTRAL_VALUE
    d1, d2, d3 = transformed_addition.shape
    output[:d1,:d2,:d3] = transformed_addition[:]
    # Return the correctly sized addition image
    return output

# A minimization objective function that evaluates the ability of an
# adversarial image 
def turn_left(addition):
    # Change the type of addition to match necessary image type
    addition = np.asarray(addition, dtype=np.uint8)
    # Reshape the addition to be the same shape as the image
    addition = addition.reshape(ADVERSARIAL_IMAGE_SIZE)
    # Collect all the deltas for this addition
    all_delta = []
    # print(addition.shape)
    addition = Image.fromarray(addition)
    # addition.save("initial_addition.png")
    # Cycle N random transformations
    for transform in range(NUM_TRANSFORMATIONS):
        print("[%s>%s]"%("="*int(round(50*transform/NUM_TRANSFORMATIONS)),
                         " "*int(50 - round(50*transform/NUM_TRANSFORMATIONS))),end="\r")
        transformed_addition = generate_transformed_addition(addition)
        # Identify the change in turning angle provided by the image
        turn = float(model.predict(np.array([transformed_addition]), batch_size=1))
        all_delta.append(turn)
        img = Image.fromarray(np.asarray(transformed_addition.reshape(IMAGE_SHAPE), dtype=np.uint8))
        img_name = "left_adv_imgs/left_adversarial_%i(%.2f).png"%(transform, turn)
        img.save(img_name)

        # # Cycle all images
        # random.shuffle(images_and_steering)
        # for (original, sa) in images_and_steering[:NUM_IMAGES]:
        #     original = original.reshape(IMAGE_SHAPE)
        #     # Combine the original image with the transformed addition
        #     img = original + np.where(transformed_addition >= NEUTRAL_VALUE, 
        #                               transformed_addition - NEUTRAL_VALUE, 0)
        #     img -= np.where(transformed_addition < NEUTRAL_VALUE,
        #                     transformed_addition, 0)
        #     # Identify the change in turning angle provided by the image
        #     turn = float(model.predict(np.array([img]), batch_size=1))
        #     delta = turn - sa
        #     all_delta.append(delta)
    exit()
    print()
    print(all_delta)
    # Return the average delta achieved by all transformations on all images
    return sum(all_delta) / len(all_delta)

# fake_img = np.random.randint(0,255,size=ADVERSARIAL_IMAGE_SIZE, dtype=np.uint8).flatten()
# fake_img[:] = 125
# print(fake_img.shape)
# turn_left(fake_img)

# ====================================
#      Adversarial Image Analysis     
# ====================================

with open("left_random_solution_1.pkl", "rb") as f:
    output = pickle.load(f)

from util.plotly import Plot
p = Plot()
p.add_histogram("Adversarial Image", output)
p.plot(file_name="left_random_solution_1.html", show=False)

print("Turn prouduced:", turn_left(output))
img = Image.fromarray(np.asarray(output.reshape(ADVERSARIAL_IMAGE_SIZE),
                                 dtype=np.uint8))
# img.save("left_random_solution_1.png")

exit()

# ======================================
#      Adversarial Image Generation     
# ======================================

from util.optimize import AMPGO, AdaptiveNormal, Random
from util.optimize import minimize
from best_adv import sol as initial_solution

sample_img = np.ones(ADVERSARIAL_IMAGE_SIZE, dtype=np.uint8).flatten()*NEUTRAL_VALUE
print("Starting optimization.")
bounds = [(0,255)]*np.prod(ADVERSARIAL_IMAGE_SIZE)
output = minimize(turn_left, initial_solution, bounds=bounds,
                  max_time=MAX_COMPUTE_TIME_SEC,
                  method=AdaptiveNormal, display=True)

print(output)
with open("left_random_solution_1.pkl", "wb") as f:
    pickle.dump(output, f)
