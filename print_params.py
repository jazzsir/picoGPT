import json
import os
import re

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder

def shape_tree(d):
    if isinstance(d, np.ndarray):
        return list(d.shape)
    elif isinstance(d, list):
        return [shape_tree(v) for v in d]
    elif isinstance(d, dict):
        return {k: shape_tree(v) for k, v in d.items()}
    else:
        ValueError("uh oh")

def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params




model_dir = os.path.join("models", "124M")
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

print(shape_tree(params))
