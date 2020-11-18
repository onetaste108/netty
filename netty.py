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
# tf.compat.v1.disable_eager_execution()
# tf.disable_eager_execution()

class Netty:
    def __init__(self):

        self.build_config = {
            "variational": False,
            "variational_w": 1e-1,
            "variational_pow": 1.25,

            "content": False,
            "content_w": 1,
            "content_layers": [12],

            "style": True,
            "style_w": 1,
            "style_layers": [1,4,7,12,17],
            "style_lw": [1,1,1,1,1],

            "mrf": False,
            "mrf_layers": [7],
            "mrf_patch_size": 16,
            "mrf_patch_stride": 16,
            "mrf_w": 1,

            "model": "vgg19",
            "pool": "avg",
            "padding": "valid",

            "size": [512,512],
        }

        self.render_config = {
            "size": [512,512],

            "maxfun": 100,
            "display": 25,
            "callback_fn": None,
        }

        self.model = None
        self.eval = None
        self.feed = {}
        self.modules = {}
        self.tgs = []

    def build(self):
        self.model, self.modules = build(self.build_config)
        self.eval = self.make_eval()

    def make_eval(self):
        grads = K.gradients(self.model.output, self.model.inputs[0])[0]
        outputs = [self.model.output] + [grads]
        return K.function(self.model.inputs, outputs)

    def make_callback(self):
        i = [0]
        display = self.render_config["display"]
        cbfn = self.render_config["callback_fn"]
        def fn(x0_ext):
            x0 = self.feed["x0"].copy()
            np.place(x0, self.feed["x0_mask"], x0_ext)
            x0 = x0.reshape((1, *self.feed["x0"].shape))
            outs = self.eval([x0]+self.tgs)
            loss_value = outs[0]
            grad_values = np.extract(self.feed["x0_mask"], np.array(outs[1:]).flatten().astype('float64'))



            if cbfn is not None:
                cbfn(i[0],deprocess(x0[0]))

            if i[0] % 10 == 0 and i[0] != 0:
                print(i[0],end=" ")
            else:
                print(".",end=" ")

            if i[0] % display == 0 and i[0] != 0:
                im.show(deprocess(x0[0]))
            i[0] += 1

            return loss_value, grad_values
        return fn

    def render(self):
        x0 = self.feed["x0"]
        x0_ext = np.extract(self.feed["x0_mask"], x0.ravel())
        callback = self.make_callback()
        bounds = get_bounds(x0_ext)
        x0_ext, min_val, info = fmin_l_bfgs_b(callback, x0_ext, bounds=bounds, maxfun=self.render_config["maxfun"])
        x = self.feed["x0"].copy()
        np.place(x, self.feed["x0_mask"], x0_ext)
        x = x.reshape(self.feed["x0"].shape)
        x = deprocess(x)
        im.show(x)
        print("Rendered!")
        return x

    def setup(self):
        tgs = []
        if self.build_config["content"]:
            t = self.get_content_tgs()
            tgs.append(t)
        if self.build_config["style"]:
            mask = self.feed["_x0_mask"]
            if type(mask) is list: tgs.extend(mask)
            else: tgs.append(mask)
                
            t = self.get_style_tgs()
            if type(t) is list: tgs.extend(t)
            else: tgs.append(t)
        if self.build_config["mrf"]:
            t = self.get_mrf_tgs()
            tgs.append(t)
        self.set_tgs(tgs)

    def get_style_tgs(self):
        n = len(self.feed["style"])
        tgs = []
        for i in range(n):
            t = self.modules["style"].predict([np.array([self.feed["style"][i]])]+self.feed["style_masks"][i])
            tgs.append(t)
        return nutil.mix_tgs(tgs,self.render_config["style_imgs_w"])

    def get_content_tgs(self):
        tgs = self.modules["content"].predict(np.array([self.feed["content"]]))
        if type(tgs) != list:
            tgs = [tgs]
        tgs.extend(self.feed["content_masks"])
        return tgs

    def get_mrf_tgs(self):
        tgs = self.modules["mrf"].predict(np.array(self.feed["style"]))
        return tgs

    def set_tgs(self,tgs):
        self.tgs = tgs

    def set_style(self,imgs,masks=None,scales=None,w=None):
        self.render_config["style_imgs_w"] = w
        if type(imgs) is not list: imgs = [imgs]

        if masks is None:
            masks = [None for i in range(len(imgs))]
        else:
            if type(masks) is not list: masks = [masks]

        if scales is None:
            scales = [1 for i in range(len(imgs))]
        else:
            if type(scales) is not list:
                scales = [scales for i in range(len(imgs))]

        self.feed["style"] = []
        self.feed["style_masks"] = []

        for img, mask, scale in zip(imgs,masks,scales):
            if scale == 0:
                factor = 1
            else:
                factor=im.propscale(img.shape[:2],self.render_config["size"][::-1]) * scale
            img = preprocess(im.size(img, factor=factor))
            self.feed["style"].append(img)

            if mask is not None:
                mask = im.size(mask,img.shape[:2][::-1])
                l_mask = scale_mask(mask,self.build_config["style_layers"])
            else:
                l_mask = []
                for l in self.build_config["style_layers"]:
                    vgg_shape = get_vgg_shape(img.shape[:2],l)[:-1]
                    l_mask.append(np.ones([1,vgg_shape[0],vgg_shape[1]],np.float32))
            self.feed["style_masks"].append(l_mask)

    def set_content(self,img,mask=None):
        img = preprocess(im.size(img, self.render_config["size"]))
        if mask is not None:
            l_mask = scale_float_mask(mask,self.build_config["content_layers"])
        else:
            l_mask = []
            for l in self.build_config["content_layers"]:
                vgg_shape = get_vgg_shape(img.shape[:2],l)[:-1]
                l_mask.append(np.ones([1,vgg_shape[0],vgg_shape[1]],np.float32))
        self.feed["content"] = img
        self.feed["content_masks"] = l_mask

    def set_x0(self,img=None,mask=None,_mask=None):
        if img is None:
            self.feed["x0"] = np.random.randn(self.render_config["size"][1],self.render_config["size"][0],3) * 10
        else:
            img = preprocess(im.size(img, self.render_config["size"]))
            self.feed["x0"] = img

        if mask is None:
            mask = np.ones([self.render_config["size"][1],self.render_config["size"][0]],np.float32)
        else:
            mask = np.float32(mask/255)
        mask = np.repeat(mask, 3).ravel()
        self.feed["x0_mask"] = mask > 0.5

        if _mask is not None:
            _mask = im.size(_mask,self.render_config["size"])
            l_mask = scale_mask(_mask,self.build_config["style_layers"])
        else:
            l_mask = []
            for l in self.build_config["style_layers"]:
                vgg_shape = get_vgg_shape(self.render_config["size"],l)[:-1]
                l_mask.append(np.ones([1,vgg_shape[1],vgg_shape[0]],np.float32))
        self.feed["_x0_mask"] = l_mask


    def clear(self):
        K.clear_session()
