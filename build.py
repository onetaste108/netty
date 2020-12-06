from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from netty import model_vgg
from netty import model_variational
from netty import module_content
from netty import module_style
from netty import module_mrf

def build(args):
    input = Input((None,None,3))

    losses = []
    module_inputs = []
    modules = {}

    for a in args:

        if a["type"] == "variational":

            loss_model = model_variational.build(a)
            losses.append(loss_model(input))

        elif a["type"] == "content":

            loss_model, target_model, targets = module_content.build(a)
            losses.append(loss_model([input] + targets))
            module_inputs.extend(targets)
            a["module"] = target_model

        elif a["type"] == "style":

            loss_model, target_model, targets = module_style.build(a)
            losses.append(loss_model([input] + targets))
            module_inputs.extend(targets)
            a["module"] = target_model

        elif a["type"] == "mrf": 

            loss_model, target_model, targets = module_mrf.build(args)
            losses.append(loss_model([input] + targets))
            module_inputs.extend(targets)
            a["module"] = target_model

    if len(losses) == 0:
        print("Nothing to optimize")
        return None

    loss = Lambda(lambda x: K.expand_dims(K.sum(x) / len(losses)))(losses)
    model = Model([input] + module_inputs, loss)
    return model
