import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def load_graph(models_dir, model_name, temperature=0.8, top_k=40,top_p=1,batch_size=1, length=None, seed = None):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        return enc, output, ckpt, hparams

def predict(enc, output, ckpt, hparams, batch_size=1):
    with tf.Session(graph=tf.Graph()) as sess:

        out = sess.run(output)
        for i in range(batch_size):
            text = enc.decode(out[i])
            return text
