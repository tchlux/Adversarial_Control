import pickle
import numpy as np
from PIL import Image
from image import random_transformation
from keras.models import load_model

#      Load the images and correct steering angles     
# =====================================================
DATA_FILE = "images_and_steering_angles.pkl"
with open(DATA_FILE, "rb") as f:
    IMAGES_AND_STEERING = pickle.load(f)
# Load the keras model for driving     
MODEL = load_model("final_model.h5")

#        Global variables
# ==============================
IMAGE_SHAPE = IMAGES_AND_STEERING[0][0].shape[1:]
RANDOM_SEED = 0
ADVERSARIAL_IMAGE_SIZE = tuple(np.array(IMAGE_SHAPE[:-1]) // 3)+(IMAGE_SHAPE[-1],)
ADVERSARIAL_IMAGE_SCALE = [-1,3]
NEUTRAL_VALUE = 125
NUM_TRANSFORMATIONS = 100
NUM_IMAGES = 50
MAX_COMPUTE_TIME_SEC = ((60*60)*24)
PROGRESS_LEN = 50

#      Program Control Flow     
# ==============================
SAVE_ALL_IMAGES = True
NEUTRAL_IMAGE_TEST = False
TRAIN_ON_REAL_IMAGES = False
PLOT_DELTA_DISTRIBUTION = False
ANALYZE_EXISTING_IMAGE = False
GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES = True
GENERATE_CLEVERHANS_ADVERSARIAL_IMAGES = not GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES
TURN_NAME = "left"

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
def turn(addition, turn_name=TURN_NAME):
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
        print("[{:s}>{:s}]".format("-"*int(round(PROGRESS_LEN*transform/NUM_TRANSFORMATIONS)),
                                   " "*int(PROGRESS_LEN - round(PROGRESS_LEN*transform/NUM_TRANSFORMATIONS))),end="\r")
        if TRAIN_ON_REAL_IMAGES:
            # Cycle all images real training images
            random.shuffle(IMAGES_AND_STEERING)
            for (original, sa) in IMAGES_AND_STEERING[:NUM_IMAGES]:
                original = original.reshape(IMAGE_SHAPE)
                # Combine the original image with the transformed addition
                img = original + np.where(transformed_addition >= NEUTRAL_VALUE, 
                                          transformed_addition - NEUTRAL_VALUE, 0)
                img -= np.where(transformed_addition < NEUTRAL_VALUE,
                                transformed_addition, 0)
                # Identify the change in turning angle provided by the image
                turn = float(MODEL.predict(np.array([img]), batch_size=1))
                delta = turn - sa
                all_delta.append(delta)
                if SAVE_ALL_IMAGES:
                    img = Image.fromarray(np.asarray(img.reshape(IMAGE_SHAPE), dtype=np.uint8))
                    img_name = "{turn}_adv_imgs/({angle:+.2f})_{turn}_adversarial_{num:03d}-{orig:+.2f}_NEUTRAL.png".format(
                        turn=turn_name, angle=turn, num=transform, orig=sa)
                    img.save(img_name)
        else:
            # Train on the ability to turn
            transformed_addition = generate_transformed_addition(addition)
            # Identify the change in turning angle provided by the image
            turn = float(MODEL.predict(np.array([transformed_addition]), batch_size=1))
            all_delta.append(turn)
            if SAVE_ALL_IMAGES:
                img = Image.fromarray(np.asarray(transformed_addition.reshape(IMAGE_SHAPE), dtype=np.uint8))
                img_name = "{turn}_adv_imgs/({angle:+.2f})_{turn}_adversarial_{num:03d}_NEUTRAL.png".format(
                    turn=turn_name, angle=turn, num=transform)
                img.save(img_name)

    print()

    if SAVE_ALL_IMAGES: 
        print("Done saving first round of images.")
        exit()

    if PLOT_DELTA_DISTRIBUTION:
        print(all_delta)
        from util.plotly import Plot
        p = Plot("Randomly Transformed Adversarial {:s} Turn Image (100 bins)".format(turn_name.title()), 
                 "Normalized Turning Angle", "Probability")
        p.add_histogram("Turn Angles", all_delta)
        p.plot(show=False, file_name="{}_adversarial_turn_angles.html".format(turn_name),show_legend=False)

    # Return the average delta achieved by all transformations on all images
    return sum(all_delta) / len(all_delta)

if NEUTRAL_IMAGE_TEST:
    fake_img = np.random.randint(0,255,size=ADVERSARIAL_IMAGE_SIZE, dtype=np.uint8).flatten()
    fake_img[:] = 125
    print(fake_img.shape)
    print(turn(fake_img))

# ====================================
#      Adversarial Image Analysis     
# ====================================
if ANALYZE_EXISTING_IMAGE:
    with open("{}_random_solution_1.pkl".format(TURN_NAME), "rb") as f:
        output = pickle.load(f)

    from util.plotly import Plot
    p = Plot()
    p.add_histogram("Adversarial Image", output)
    p.plot(file_name="{}_random_solution_1.html".format(TURN_NAME), show=False)

    print("Turn prouduced:", turn(output))
    img = Image.fromarray(np.asarray(output.reshape(ADVERSARIAL_IMAGE_SIZE), dtype=np.uint8))
    img.save("{}_random_solution_1.png".format(TURN_NAME))
    exit()

# ======================================
#      Adversarial Image Generation     
# ======================================

if GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES:
    from util.optimize import AMPGO, AdaptiveNormal, Random
    from util.optimize import minimize
    if TURN_NAME == "left":
        from left_random_solution_1 import sol as initial_solution
    else:
        from right_random_solution_1 import sol as initial_solution

    sample_img = np.ones(ADVERSARIAL_IMAGE_SIZE, dtype=np.uint8).flatten()*NEUTRAL_VALUE
    print("Starting optimization.")
    bounds = [(0,255)]*np.prod(ADVERSARIAL_IMAGE_SIZE)
    output = minimize(turn, initial_solution, bounds=bounds,
                      max_time=MAX_COMPUTE_TIME_SEC,
                      method=AdaptiveNormal, display=True)

    print(output)
    with open("{}_random_solution_1.pkl".format(TURN_NAME), "wb") as f:
        pickle.dump(output, f)


if GENERATE_CLEVERHANS_ADVERSARIAL_IMAGES:
    pass
