import pickle, sys, random
import numpy as np
from PIL import Image
from image import random_transformation
stdout, stderr = sys.stdout, sys.stderr
with open("/dev/null","w") as f:
    sys.stdout = sys.stderr = f
    from keras.models import load_model
    sys.stdout = stdout
    sys.stderr = stderr

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
NUM_IMAGES = 1
MAX_COMPUTE_TIME_SEC = ((60*1)*60) # ((60*60)*24) # ((min*sec)*hours)
PROGRESS_LEN = 50

#      Program Control Flow     
# ==============================
SAVE_ALL_IMAGES = True
NEUTRAL_IMAGE_TEST = False
TRAIN_ON_REAL_IMAGES = False
PLOT_DELTA_DISTRIBUTION = False
ANALYZE_EXISTING_IMAGE = False
USE_RANDOM_TRANSFORMATIONS = True
GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES = True
GENERATE_CLEVERHANS_ADVERSARIAL_IMAGES = not GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES
TURN_NAME = "left"
SOLUTION_NUMBER = "1"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===================================================
#      Conditional Imports Based on Control Flow     
# ===================================================
if PLOT_DELTA_DISTRIBUTION or ANALYZE_EXISTING_IMAGE:
    from util.plotly import Plot
if GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES:
    from util.optimize import AMPGO, AdaptiveNormal, Random
    from util.optimize import minimize
    if TURN_NAME == "left":
        from left_random_solution_1 import solution as initial_solution
    else:
        from right_random_solution_2 import solution as initial_solution
if GENERATE_CLEVERHANS_ADVERSARIAL_IMAGES:
    from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod


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
    if USE_RANDOM_TRANSFORMATIONS:
        # Cycle N random transformations
        for transform in range(NUM_TRANSFORMATIONS):
            print("[{:s}>{:s}]".format("-"*int(round(PROGRESS_LEN*transform/NUM_TRANSFORMATIONS)),
                                       " "*int(PROGRESS_LEN - round(PROGRESS_LEN*transform/NUM_TRANSFORMATIONS))),end="\r")
            # Train on the ability to turn
            transformed_addition = generate_transformed_addition(addition)
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
                        img_name = "{turn}_adv_imgs/({angle:+.2f})_{turn}_adversarial_{num:03d}-({orig:+.2f}).png".format(
                            turn=turn_name, angle=turn, num=transform, orig=sa)
                        img.save(img_name)
                        print("Saved '%s'"%img_name)
            else:
                # Identify the change in turning angle provided by the image
                turn = float(MODEL.predict(np.array([transformed_addition]), batch_size=1))
                all_delta.append(turn)
                if SAVE_ALL_IMAGES:
                    img = Image.fromarray(np.asarray(transformed_addition.reshape(IMAGE_SHAPE), dtype=np.uint8))
                    img_name = "{turn}_adv_imgs/({angle:+.2f})_{turn}_adversarial_{num:03d}_NEUTRAL.png".format(
                        turn=turn_name, angle=turn, num=transform)
                    img.save(img_name)
                    print("Saved '%s'"%img_name)
        print()
    else:
        # Do not use random transformations, just optimize over a
        # single image that takes up the entire view field of the car.
        addition = addition.resize(IMAGE_SHAPE[:-1])
        adv_img = np.array(addition).reshape((1,)+IMAGE_SHAPE)
        all_delta = [float(MODEL.predict(adv_img))]

    if SAVE_ALL_IMAGES: 
        print("Done saving first round of images.")
        exit()

    if PLOT_DELTA_DISTRIBUTION:
        print(all_delta)
        p = Plot("Randomly Transformed Adversarial {:s} Turn Image (100 bins)".format(turn_name.title()), 
                 "Normalized Turning Angle", "Probability")
        p.add_histogram("Turn Angles", all_delta)
        p.plot(show=False, file_name="{}_adversarial_turn_angles.html".format(turn_name),show_legend=False)

    # Flip the sign if we are optimizing for right turns
    avg_delta = sum(all_delta) / len(all_delta)
    if TURN_NAME == "right": avg_delta = -avg_delta
    # Return the average delta achieved by all transformations on all images
    return avg_delta

if NEUTRAL_IMAGE_TEST:
    fake_img = np.random.randint(0,255,size=ADVERSARIAL_IMAGE_SIZE, dtype=np.uint8).flatten()
    fake_img[:] = 125
    print(fake_img.shape)
    print(turn(fake_img))

# ====================================
#      Adversarial Image Analysis     
# ====================================
if ANALYZE_EXISTING_IMAGE:
    # with open("{}_random_solution_{}.pkl".format(TURN_NAME, SOLUTION_NUMBER), "rb") as f:
    #     output = pickle.load(f)
    output = np.array(initial_solution)

    p = Plot()
    p.add_histogram("Adversarial Image", output)
    p.plot(file_name="{}_random_solution_{}.html".format(TURN_NAME, SOLUTION_NUMBER), show=False)

    print("Turn prouduced:", turn(output))
    img = Image.fromarray(np.asarray(output.reshape(ADVERSARIAL_IMAGE_SIZE), dtype=np.uint8))
    img.save("{}_random_solution_{}.png".format(TURN_NAME, SOLUTION_NUMBER))
    exit()

# ======================================
#      Adversarial Image Generation     
# ======================================
if GENERATE_OPTIMIZED_ADVERSARIAL_IMAGES:
    sample_img = np.ones(ADVERSARIAL_IMAGE_SIZE, dtype=np.uint8).flatten()*NEUTRAL_VALUE
    print("Starting optimization.")
    bounds = [(0,255)]*np.prod(ADVERSARIAL_IMAGE_SIZE)
    checkpoint_file = "{}_random_solution_{}.py".format(TURN_NAME, SOLUTION_NUMBER)
    output = minimize(turn, initial_solution, bounds=bounds,
                      max_time=MAX_COMPUTE_TIME_SEC,
                      method=AdaptiveNormal, display=True,
                      checkpoint_file=checkpoint_file)
    print(output)
    with open("{}_random_solution_2.pkl".format(TURN_NAME), "wb") as f:
        pickle.dump(output, f)


if GENERATE_CLEVERHANS_ADVERSARIAL_IMAGES:

    # (img, steer_ang) = IMAGES_AND_STEERING[0]
    # img = np.array([img.reshape(IMAGE_SHAPE)])
    # attack = CarliniWagnerL2.generate(MODEL, img)
    # print()
    # help(attack)
    # print()
    # print(attack)

    # ================================
    #      Cleverhans attack code     
    # ================================

    # from __future__ import absolute_import
    # from __future__ import division
    # from __future__ import print_function
    # from __future__ import unicode_literals

    import numpy as np
    import keras
    from keras import backend
    from keras.optimizers import Adam
    from keras.models import Sequential
    from keras.layers import Lambda, Activation
    import tensorflow as tf
    from tensorflow.python.platform import flags

    from cleverhans.utils_tf import model_train, model_eval
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils import AccuracyReport
    from cleverhans.utils_keras import cnn_model
    from cleverhans.utils_keras import KerasModelWrapper

    FLAGS = flags.FLAGS

    # tf.app.run()
    train_start=0
    train_end=60000
    test_start=0
    test_end=10000
    nb_epochs=6
    batch_size=128
    learning_rate=0.001
    train_dir="/tmp"
    filename="mnist.ckpt"
    testing=False
    # keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Load the keras model for driving (in current session)
    model_with_softmax = load_model("final_model.h5")

    # Get the x values for identifying adversarial samples
    x = np.array([img[0] for (img,sa) in IMAGES_AND_STEERING])

    # Generate a model that adds a softmax output to the original model
    def act_func(turn_angle):
        # Use this activation function before the softmax (to
        # 'classify' left and right turns). This will return larger
        # numbers in index[0] for left turns.
        mat = tf.constant([[-1.0, 1.0]])
        result = tf.matmul(turn_angle, mat)
        return result
    model_with_softmax.add(Lambda(act_func, name="expand_to_classify"))
    model_with_softmax.add(Activation("softmax"))
    model_with_softmax.compile(loss='mean_squared_error', optimizer=Adam(lr=10**(-4)))
    # Print out a summary of the model being used
    # model_with_softmax.summary()

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(MODEL)

    # ===============================================================
    # WARNING: FAILURE POINT. Beyond this, the code fails. Neither of
    # MODEL nor model_with_softmax have computable gradients by
    # TensorFlow, which causes the Cleverhans code to fail.
    # ===============================================================

    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    # Consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = model(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Calculating train error
    if testing:
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_train,
                         Y_train, args=eval_par)
        report.train_clean_train_adv_eval = acc

    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = cnn_model()
    preds_2 = model_2(x)
    wrap_2 = KerasModelWrapper(model_2)
    fgsm2 = FastGradientMethod(wrap_2, sess=sess)
    preds_2_adv = model_2(fgsm2.generate(x, **fgsm_params))

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, save=False, rng=rng)

    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    print(report)
