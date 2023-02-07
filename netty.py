import numpy as np
from netty.build import build
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import backend as K
import netty.imutil as im
from netty.vgg_utils import *
from netty import gram_patcher
from netty import netty_utils as nutil
tf.compat.v1.disable_eager_execution()
# tf.disable_eager_execution()
from tqdm import tqdm

class Netty:
    def __init__(self):
        self.args = {
            "size": [256, 256],
            "iters": 100,
            "modules": [],

            "model": "vgg19",
            "pool": "avg",
            "padding": "valid"
        }

        self.iteration = 0
        self.model = None
        self.eval = None
        self.feed = {}
        self.modules = {}
        self.tgs = []
        self.display_every = 25
        self.clear()

    def build(self):
        self.model = build(self.args["modules"])
        self.eval = self.make_eval()

    def make_eval(self):
        grads = K.gradients(self.model.output, self.model.inputs[0])[0]
        outputs = [self.model.output, tf.cast(grads, tf.float64)]
        return K.function(self.model.inputs, outputs)

    def callback(self, x, i):
        if (i % self.display_every == 0):
            im.clear()
            im.show(np.clip(deprocess(x),0,255))

    def optimize(self, x):
        xfull = self.xfull
        np.place(xfull, self.xmask, x)
        self.callback(xfull, self.iteration)
        self.bar.update(1)
        xfull = np.expand_dims(xfull, 0)
        loss, grad = self.eval([xfull, *self.tgs])
        grad = np.extract(self.xmask, grad)
        self.iteration += 1
        return loss, grad.astype(np.float64)

    def _render(self):
        self.xfull = self.feed["x0"].copy()
        x = np.extract(self.xmask, self.xfull)
        bounds = get_bounds(np.reshape(x, (-1,3)))
        self.iteration = 0
        self.bar = tqdm(total=self.args["iters"], position=0, leave=True, ncols=100)
        x, min_val, info = fmin_l_bfgs_b(self.optimize, x, bounds=bounds, maxfun=self.args["iters"])
        self.bar.close()
        np.place(self.xfull, self.xmask, x)
        out = deprocess(self.xfull)
        im.clear()
        im.show(out)
        print("Render finished")
        return out

    def render(self, iters=100):
        self.args["iters"] = iters
        self.build()
        self.get_targets()
        return self._render()
        

    def get_targets(self):
        args = self.args["modules"]
        tgs = []
        for a in args:
            if a["type"] == "content":
                target = a["module"].predict([a["target"]])
                if not type(target) == list: target = [target]
                t = target + a["masks"]
                tgs.append(t)
            elif a["type"] == "style":
                target = a["module"].predict([a["target"], *a["masks"]])
                if not type(target) == list: target = [target]
                t = a["x_mask"] + target
                tgs.append(t)
            elif a["type"] == "mrf":
                t = a["module"].predict([a["target"]])
                tgs.append(t)
        self.tgs = tgs

    def size(self, *size):
        self.args["size"] = size

    def display(self, val):
        self.display_every = val

    def style(self, img, layers=[1,4,7], layer_weights=None, scale=1, weight=1, mask=None, x_mask=None):
        module = {"type": "style", "weight": weight}
        module["name"] = "style" + str(self.nmodules["style"])
        self.nmodules["style"] += 1
        module["layers"] = layers
        module["layers_weights"] = [1 for l in layers] if layer_weights is None else layer_weights
        module["target"] = np.expand_dims(preprocess(im.size(img, factor=scale*im.propscale(img.shape[:2], self.args["size"][::-1])) if scale is not None else img), 0)
        module["x_mask"] = scale_float_mask(np.ones(self.args["size"][::-1]) if x_mask is None else x_mask, layers)
        module["masks"] = scale_float_mask(np.ones(module["target"].shape[1:3]) if mask is None else mask, layers)
        self.args["modules"].append(module)

    def content(self, img, layers=[12], layer_weights=None, weight=1, mask=None):
        module = {"type": "content", "weight": weight}
        module["name"] = "content" + str(self.nmodules["content"])
        self.nmodules["content"] += 1
        module["layers"] = layers
        module["layers_weights"] = [1 for l in layers] if layer_weights is None else layer_weights
        module["target"] = np.expand_dims(preprocess(im.size(img, self.args["size"])), 0)
        module["masks"] = scale_float_mask(np.ones(module["target"].shape[1:3]) if mask is None else mask, layers)
        self.args["modules"].append(module)

    def mrf(self, img, layers=[4], layers_weights=None, patch_size=3, patch_stride=1):
        module = {"type": "mrf", "weight": weight}
        module["name"] = "mrf" + str(self.nmodules["mrf"])
        self.nmodules["mrf"] += 1
        module["layers"] = layers
        module["layers_weights"] = [1 for l in layers] if layer_weights is None else layer_weights
        module["target"] = im.size(img, factor=im.propscale(img.shape[:2], self.args["size"][::-1]))
        module["patch_size"] = patch_size
        module["patch_stride"] = patch_stride
        module["shape"] = self.args["size"][::-1]
        module["target_shape"] = module["target"].shape[:2]
        self.args["modules"].append(module)

    def x0(self, x0=None, mask=None):
        self.feed["x0"] = preprocess(im.size(x0, self.args["size"]) if x0 is not None else 128+nutil.noise(self.args["size"]))
        self.xmask = mask if mask is not None else np.ones(self.args["size"][::-1], dtype=bool)
        self.xmask = np.repeat(np.expand_dims(self.xmask, -1), 3, axis=-1)

    def var(self, weight=1, power=0.25):
        module = {"type": "variational", "weight": weight, "power": power}
        module["name"] = "variational" + str(self.nmodules["variational"])
        self.args["modules"].append(module)

    def clear(self):
        self.model = None
        self.eval = None
        K.clear_session()
        self.args["modules"] = []
        self.nmodules = {
            "variational": 0, "content": 0, "style": 0, "mrf": 0
        }






    def build_style_map_model(self, layers=[1,4,7], smooth=1, mean=1):
        import tensorflow as tf
        from netty import module_style as ms
        from netty import model_variational as mv
        from tensorflow.keras.layers import Input, Lambda
        from tensorflow.keras.models import Model

        tf.disable_eager_execution()
        tf.keras.backend.clear_session()

        vgg = ms.model_vgg.build({})
        vgg = ms.extract_layers(vgg, layers)
        mask_input = Input((None, None), name="MINPUT")
        style_input = Input((None, None, 3), name="SINPUT")
        render_input = Input((None, None, 3), name="RINPUT")

        def resize_fn(args):
            x = args[0]
            y = args[1]
            return tf.compat.v2.image.resize(tf.expand_dims(x,-1), tf.shape(y[0])[:2], antialias=True)[...,0]
        resize = Lambda(resize_fn)
        apmask = ms.apply_mask()
        loss_layer = ms.loss_l(1/len(vgg.outputs))
        gram_layer = ms.gram_l(0)
        mask_gram_layer = ms.mask_gram_l()

        vgg_a = vgg(style_input)
        vgg_b = vgg(render_input)


        losses = []
        for a, b in zip(vgg_a, vgg_b):
            mask = resize([mask_input, a])
            a = apmask([a, mask])
            loss = loss_layer([gram_layer(a), gram_layer(b)])
            losses.append(loss)
        loss = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(x), 0))(losses)

        var = mv.build({"weight": smooth, "power": 1.25})
        var_loss = var(Lambda(lambda x: tf.expand_dims(x, -1))(mask_input))
        loss = Lambda(lambda x: x[0]+x[1])([loss, var_loss])
        mean_loss = Lambda(lambda x: tf.square(tf.reduce_mean(x)-1)*mean)(mask_input)
        loss = Lambda(lambda x: x[0]+x[1])([loss, mean_loss])
        self.style_map_model = Model([mask_input, style_input, render_input], [loss])

    def optimize_style_map(self, patch, style, iters=500):
        model = self.style_map_model
        def make_eval():
            grads = tf.keras.backend.gradients(model.output, model.inputs[0])[0]
            outputs = [model.output, tf.cast(grads, tf.float64)]
            return tf.keras.backend.function(model.inputs, outputs)
        ev = make_eval()
        i = [0]
        # im.show(style)
        # im.show(patch)
        style = np.float32([preprocess(style)])
        patch = np.float32([preprocess(patch)])

        xmask = np.ones(style.shape[:3])
        bar = tqdm(total=iters, position=0, leave=True, ncols=100)
        def optimize(x):
            x = np.reshape(x, style.shape[:3])
            if (i[0]%50 == 0):
                im.clear()
                im.show(im.norm(x[0])*255)
            loss, grad = ev([x, style, patch])
            # loss = loss / 1000
            bar.set_postfix(loss=loss[0])
            bar.update(1)
            i[0] += 1
            return loss, grad.flatten()
        x = xmask.flatten()
        bounds = np.repeat(np.float32([[0,1000000000]]), x.size, axis=0)
        x, min_val, info = fmin_l_bfgs_b(optimize, x, bounds=bounds, maxfun=iters)
        newmask = np.reshape(x, style.shape[1:3])
        bar.close()
        print(info)
        return newmask

    # def set_style(self,imgs,masks=None,scales=None,w=None):
    #     self.render_config["style_imgs_w"] = w
    #     if type(imgs) is not list: imgs = [imgs]

    #     if masks is None:
    #         masks = [None for i in range(len(imgs))]
    #     else:
    #         if type(masks) is not list: masks = [masks]

    #     if scales is None:
    #         scales = [1 for i in range(len(imgs))]
    #     else:
    #         if type(scales) is not list:
    #             scales = [scales for i in range(len(imgs))]

    #     self.feed["style"] = []
    #     self.feed["style_masks"] = []

    #     for img, mask, scale in zip(imgs,masks,scales):
    #         if scale == 0:
    #             factor = 1
    #         else:
    #             factor=im.propscale(img.shape[:2],self.render_config["size"][::-1]) * scale
    #         img = preprocess(im.size(img, factor=factor))
    #         self.feed["style"].append(img)

    #         if mask is not None:
    #             mask = im.size(mask,img.shape[:2][::-1])
    #             l_mask = scale_mask(mask,self.build_config["style_layers"])
    #         else:
    #             l_mask = []
    #             for l in self.build_config["style_layers"]:
    #                 vgg_shape = get_vgg_shape(img.shape[:2],l)[:-1]
    #                 l_mask.append(np.ones([1,vgg_shape[0],vgg_shape[1]],np.float32))
    #         self.feed["style_masks"].append(l_mask)

    # def set_content(self,img,mask=None):
    #     img = preprocess(im.size(img, self.render_config["size"]))
    #     if mask is not None:
    #         l_mask = scale_float_mask(mask,self.build_config["content_layers"])
    #     else:
    #         l_mask = []
    #         for l in self.build_config["content_layers"]:
    #             vgg_shape = get_vgg_shape(img.shape[:2],l)[:-1]
    #             l_mask.append(np.ones([1,vgg_shape[0],vgg_shape[1]],np.float32))
    #     self.feed["content"] = img
    #     self.feed["content_masks"] = l_mask

    # def set_x0(self,img=None,mask=None,_mask=None):
    #     if img is None:
    #         self.feed["x0"] = np.random.randn(self.render_config["size"][1],self.render_config["size"][0],3) * 10
    #     else:
    #         img = preprocess(im.size(img, self.render_config["size"]))
    #         self.feed["x0"] = img

    #     if mask is None:
    #         mask = np.ones([self.render_config["size"][1],self.render_config["size"][0]],np.float32)
    #     else:
    #         mask = np.float32(mask/255)
    #     mask = np.repeat(mask, 3).ravel()
    #     self.feed["x0_mask"] = mask > 0.5

    #     if _mask is not None:
    #         _mask = im.size(_mask,self.render_config["size"])
    #         l_mask = scale_mask(_mask,self.build_config["style_layers"])
    #     else:
    #         l_mask = []
    #         for l in self.build_config["style_layers"]:
    #             vgg_shape = get_vgg_shape(self.render_config["size"],l)[:-1]
    #             l_mask.append(np.ones([1,vgg_shape[1],vgg_shape[0]],np.float32))
    #     self.feed["_x0_mask"] = l_mask

