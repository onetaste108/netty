from netty.build_utils import *
from netty import model_vgg
from netty import model_octave
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf

# def apply_mask():
#     def fn(x):
#         x1 = x[0][0]
#         x2 = x[1][0]
#         x2 = K.expand_dims(x2,-1)
#         x1 = tf.multiply(x1,x2)
#         return K.expand_dims(x1,axis=0)
#     return Lambda(fn)

def build_mask_model(model):
    apmask = apply_mask()
    mask_gram_layer = mask_gram_l()
    mask_inputs = []
    mask_outs = []
    for i,o in enumerate(model.outputs):
        mask_input = Input((o.shape[1],o.shape[2]), name="content_mask_input_"+str(i))
        mask_inputs.append(mask_input)
        mask_output = apmask([o,mask_input])
        mask_output = mask_gram_layer([mask_output,mask_input])
        mask_outs.append(mask_output)
    mask_model = Model(model.inputs+mask_inputs,mask_outs)
    return mask_model, mask_inputs

def loss_l(w):
    def fn(x):
        x = K.sum(K.square(x[0]-x[1])) * w / 2
        return K.expand_dims(x)
    return Lambda(fn)

def mask_loss_l(w):
    def fn(x):
        diff = K.square(x[0][0]-x[1][0])
        masked = diff * K.expand_dims(x[2][0], -1)
        x = K.sum(masked) * w / 2
        return K.expand_dims(x)
    return Lambda(fn)

def content_l():
    def fn(x):
        return x
    return Lambda(fn)

def build(args):
    vgg = model_vgg.build(args)

    model = extract_layers(vgg, args["layers"])

    content_layer = content_l()
    model = attach_models(model, content_layer)

    targets = []
    masks = []
    losses = []
    for i in range(len(model.outputs)):
        targets.append(Input(model.outputs[i].shape[1:], name="content_target_input_{}".format(i)))
        masks.append(Input(model.outputs[i].shape[1:-1], name="content_mask_input_{}".format(i)))
        layer_weight = 1
        layer_loss = mask_loss_l(layer_weight)([targets[i], model.outputs[i], masks[i]])
        losses.append(layer_loss)
    loss = Lambda(lambda x: K.expand_dims(K.sum(x)) / len(losses) * args["weight"])(losses)

    loss_model = Model(model.inputs + targets + masks, loss)

    return loss_model, model, targets + masks
