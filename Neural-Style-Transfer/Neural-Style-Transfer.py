import time
import numpy as np
import imageio

from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.preprocessing.image import load_img

from scipy.optimize import fmin_l_bfgs_b

# preprocessing
from utils import preprocess_image, deprocess_image

params = {
'base_img_path' : 'inputs/chicago.jpg',
'style_img_path' : 'inputs/great_wave_of_kanagawa.jpg',
'output_img_path' : 'results/wave_chicago',
'convnet' : 'vgg16',
'content_weight' : 500,
'style_weights' : (10, 10, 50, 10, 10),
'tv_weight' : 200,
'content_layer' : 'block4_conv2',
'style_layers' : ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1',
                  'block4_conv1',
                  'block5_conv1'],
'iterations' : 1
}

def feature_reconstruction_loss(base, output):
    """
    Compute the content loss for style transfer.

    Inputs:
    - output: features of the generated image, Tensor with shape [height, width, channels]
    - base: features of the content image, Tensor with shape [height, width, channels]

    Returns:
    - scalar content loss
    """
    # YOUR CODE GOES HERE
    return K.sum(K.square(output - base))

# Test your code
np.random.seed(1)
base = np.random.randn(10,10,3)
output = np.random.randn(10,10,3)
a = K.constant(base)
b = K.constant(output)
test = feature_reconstruction_loss(a, b)
print('Result:          ', K.eval(test))
print('Expected result: ', 605.62195)



def style_reconstruction_loss(base, output):
    """
    Computes the style reconstruction loss. It encourages the output img
    to have same stylistic features as style image.

    Inputs:
    - base: features at given layer of the style image.
    - output: features of the same length as base of the generated image.

    Returns:
    - style_loss: scalar style loss
    """
    # YOUR CODE GOES HERE
    H, W = int(base.shape[0]), int(base.shape[1])
    gram_base = gram_matrix(base)
    gram_output = gram_matrix(output)
    factor = 1.0 / float((2*H*W)**2)
    out = factor * K.sum(K.square(gram_output - gram_base))
    return out

def gram_matrix(x):
    """
    Computes the outer-product of the input tensor x.

    Input:
    - x: input tensor of shape (H, W, C)

    Returns:
    - tensor of shape (C, C) corresponding to the Gram matrix of
    the input image.
    """
    # YOUR CODE GOES HERE
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features))

def total_variation_loss(x):
    """
    Total variational loss. Encourages spatial smoothness
    in the output image.

    Inputs:
    - x: image with pixels, has shape 1 x H x W x C.

    Returns:
    - total variation loss, a scalar number.
    """
    # YOUR CODE GOES HERE
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.sum(a + b)

## Test your code
np.random.seed(1)
x_np = np.random.randn(1,10,10,3)
x = K.constant(x_np)
test = total_variation_loss(x)
print('Result:  ', K.eval(test))
print('Expected:', 937.0538)

def style_transfer(base_img_path, style_img_path, output_img_path, convnet='vgg16',
        content_weight=3e-2, style_weights=(20000, 500, 12, 1, 1), tv_weight=5e-2, content_layer='block4_conv2',
        style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'], iterations=50):

    print('\nInitializing Neural Style model...')

    # Determine the image sizes. Fix the output size from the content image.
    print('\n\tResizing images...')
    width, height = load_img(base_img_path).size
    new_dims = (height, width)

    # Preprocess content and style images. Resizes the style image if needed.
    content_img = K.variable(preprocess_image(base_img_path, new_dims))
    style_img = K.variable(preprocess_image(style_img_path, new_dims))

    # Create an output placeholder with desired shape.
    # It will correspond to the generated image after minimizing the loss function.
    output_img = K.placeholder((1, height, width, 3))

    # Sanity check on dimensions
    print("\tSize of content image is: {}".format(K.int_shape(content_img)))
    print("\tSize of style image is: {}".format(K.int_shape(style_img)))
    print("\tSize of output image is: {}".format(K.int_shape(output_img)))

    # Combine the 3 images into a single Keras tensor, for ease of manipulation
    # The first dimension of a tensor identifies the example/input.
    input_img = K.concatenate([content_img, style_img, output_img], axis=0)

    # Initialize the vgg16 model
    print('\tLoading {} model'.format(convnet.upper()))

    if convnet == 'vgg16':
        model = vgg16.VGG16(input_tensor=input_img, weights='imagenet', include_top=False)
    else:
        model = vgg19.VGG19(input_tensor=input_img, weights='imagenet', include_top=False)

    print('\tComputing losses...')
    # Get the symbolic outputs of each "key" layer (they have unique names).
    # The dictionary outputs an evaluation when the model is fed an input.
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Extract features from the content layer
    content_features = outputs_dict[content_layer]

    # Extract the activations of the base image and the output image
    base_image_features = content_features[0, :, :, :]  # 0 corresponds to base
    combination_features = content_features[2, :, :, :] # 2 coresponds to output

    # Calculate the feature reconstruction loss
    content_loss = content_weight * feature_reconstruction_loss(base_image_features, combination_features)

    # For each style layer compute style loss
    # The total style loss is the weighted sum of those losses
    temp_style_loss = K.variable(0.0)       # we update this variable in the loop
    weight = 1.0 / float(len(style_layers))

    for i, layer in enumerate(style_layers):
        # extract features of given layer
        style_features = outputs_dict[layer]
        # from those features, extract style and output activations
        style_image_features = style_features[1, :, :, :]   # 1 corresponds to style image
        output_style_features = style_features[2, :, :, :]  # 2 coresponds to generated image
        temp_style_loss = temp_style_loss + style_weights[i] * weight * \
                    style_reconstruction_loss(style_image_features, output_style_features)
    style_loss = temp_style_loss

    # Compute total variational loss.
    tv_loss = tv_weight * total_variation_loss(output_img)

    # Composite loss
    total_loss = content_loss + style_loss + tv_loss

    # Compute gradients of output img with respect to total_loss
    print('\tComputing gradients...')
    grads = K.gradients(total_loss, output_img)

    outputs = [total_loss] + grads
    loss_and_grads = K.function([output_img], outputs)

    # Initialize the generated image from random noise
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    # Loss function that takes a vectorized input image, for the solver
    def loss(x):
        x = x.reshape((1, height, width, 3))   # reshape
        return loss_and_grads([x])[0]

    # Gradient function that takes a vectorized input image, for the solver
    def grads(x):
        x = x.reshape((1, height, width, 3))   # reshape
        return loss_and_grads([x])[1].flatten().astype('float64')

    # Fit over the total iterations
    for i in range(iterations+1):
        print('\n\tIteration: {}'.format(i+1))

        toc = time.time()
        x, min_val, info = fmin_l_bfgs_b(loss, x.flatten(), fprime=grads, maxfun=20)

        # save current generated image
        if i%10 == 0:
            img = deprocess_image(x.copy(), height, width)
            fname = output_img_path + '_at_iteration_%d.png' % (i)
            imageio.imwrite(fname, img)
            print('\t\tImage saved as', fname)

        tic = time.time()

        print('\t\tLoss: {:.2e}, Time: {} seconds'.format(float(min_val), float(tic-toc)))

style_transfer(**params)

# Test your code
np.random.seed(1)
x = np.random.randn(10,10,3)
y = np.random.randn(10,10,3)
a = K.constant(x)
b = K.constant(y)
test = style_reconstruction_loss(a, b)
print('Result:  ', K.eval(test))
print('Expected:', 0.09799164)
