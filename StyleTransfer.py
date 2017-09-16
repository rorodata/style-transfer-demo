print("starting StyleTransfer")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import PIL
import scipy, scipy.misc, scipy.optimize
from scipy.optimize import fmin_l_bfgs_b
import io
print("importing keras")

# #import tensorflow as tf
import keras
from keras.models import Model
import keras.backend as K
from keras import metrics

print("imports done")

def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation
    # between each pair of channels
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()


def style_loss(x, targ):
    return K.mean(metrics.mse(gram_matrix(x), gram_matrix(targ)))


def StyleXfer(content_image, style_image, n_iterations=5, style_loss_wt=1.0, content_loss_wt=0.2):
    img_content=scipy.misc.imresize(content_image, (224,224))
    content_arr=img_content-127.5

    img_style=scipy.misc.imresize(style_image, (224,224))
    style_arr=img_style-127.5


    shp=(1,224,224,3)
    vgg_model=keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=shp[1:])

    outputs = {l.name: l.output for l in vgg_model.layers}

    layer=outputs['block2_conv1'] #block4_conv1
    content_model=Model(vgg_model.input, layer)

    target=content_model.predict(content_arr.reshape(shp))
    target=K.variable(target)

    loss=K.mean(metrics.mse(target, layer))
    grads=K.gradients(loss,content_model.input)

    style_layers = [outputs['block{}_conv2'.format(o)] for o in range(1,6)]
    style_model=Model(vgg_model.input, style_layers)
    style_targets=style_model.predict(style_arr.reshape(shp))
    style_targets=[K.variable(t) for t in style_targets]

    total_style_loss=sum([style_loss(i[0],j[0]) for (i,j) in zip(style_layers, style_targets)])

    #total_loss=(total_style_loss/5.0) + loss
    total_loss=(total_style_loss*style_loss_wt) + (loss*content_loss_wt)

    total_grads=K.gradients(total_loss, vgg_model.input)


    transfer_f2= K.function([vgg_model.input], [total_loss])
    transfer_f2der= K.function([vgg_model.input], total_grads)

    f2= lambda x: transfer_f2([x.reshape(shp)])[0].astype(np.float64)
    f2der=lambda x: transfer_f2der([x.reshape(shp)])[0].flatten().astype(np.float64)

    rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/1
    x = rand_img(shp)


    for i in range(n_iterations):
        print("iteration {}", i)
        x, min_val, info = fmin_l_bfgs_b(f2, x, fprime=f2der, maxfun=20)
        x = np.clip(x, -127,127)

    final_img = (x.reshape(224,224,3)+127.5).astype('uint8')

    return final_img

def style_transfer(content_fileobj, style_fileobj,
    n_iterations=5, style_loss_wt=1.0, content_loss_wt=0.2):
    """Takes a content image and a style image as file objects and
    transfers the style from style image to the content image.

    Returns the new image as a file object.
    """
    print("style_transfer", n_iterations, style_loss_wt, content_loss_wt)
    content_image = plt.imread(content_fileobj, format="jpg")
    style_image = plt.imread(style_fileobj, format="jpg")

    final_image = StyleXfer(content_image, style_image,
        n_iterations=int(n_iterations),
        style_loss_wt=float(style_loss_wt),
        content_loss_wt=float(content_loss_wt))

    f = io.BytesIO()
    plt.imsave(f, final_image)
    f.seek(0)
    return f

