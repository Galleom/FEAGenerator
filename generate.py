import os
import json
import sys
import re
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import model, sample, encoder

def start_tf_sess(threads=1, server=None):
    """
    Returns a tf.Session w/ config
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    if threads > 0:
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = threads

    if server is not None:
        return tf.compat.v1.Session(target=server.target, config=config)
    
    return tf.compat.v1.Session(config=config)


def load_gpt2(sess,
              run_name="run1",
              model_name=None,
              model_dir='models'):
    """Loads the model checkpoint or existing model into a TensorFlow session
    for repeated predictions.
    """

    checkpoint_path = os.path.join(model_dir, model_name)

    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])
    output = model.model(hparams=hparams, X=context)

    ckpt = tf.train.latest_checkpoint(checkpoint_path)
    saver = tf.compat.v1.train.Saver(allow_empty=True)
    sess.run(tf.compat.v1.global_variables_initializer())

    print('Loading pretrained model', ckpt)
    saver.restore(sess, ckpt)

def generate(sess,
             run_name='run1',
             model_name=None,
             model_dir='models',
             sample_dir='samples',
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.8,
             top_k=40,
             top_p=1.0,
             include_prefix=True):
    """Generates text from a model loaded into memory.
    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None
    checkpoint_path = os.path.join(model_dir, model_name)

    enc = encoder.get_encoder(model_name, model_dir)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if prefix:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        context_tokens = enc.encode(prefix)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams,
        length=min(length, 1023 - (len(context_tokens) if prefix else 0)),
        start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        context=context if prefix else None,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )[:, 1:]

    if destination_path:
        f = open(destination_path, 'w')
    generated = 0
    gen_texts = []
    while generated < nsamples:
        if not prefix:
            out = sess.run(output)
        else:
            out = sess.run(output, feed_dict={
                    context: batch_size * [context_tokens]
                })
        for i in range(batch_size):
            generated += 1
            gen_text = enc.decode(out[i])
            if prefix:
                gen_text = enc.decode(context_tokens[:1]) + gen_text
            if truncate:
                truncate_esc = re.escape(truncate)
                if prefix and not include_prefix:
                    prefix_esc = re.escape(prefix)
                    pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                         truncate_esc)
                else:
                    pattern = '(.*?)(?:{})'.format(truncate_esc)

                trunc_text = re.search(pattern, gen_text, re.S)
                if trunc_text:
                    gen_text = trunc_text.group(1)
            gen_text = gen_text.lstrip('\n')
            if destination_path:
                f.write("{}\n{}".format(gen_text, sample_delim))
            if not return_as_list and not destination_path:
                print("{}\n{}".format(gen_text, sample_delim), end='')
            gen_texts.append(gen_text)

    if destination_path:
        f.close()

    if return_as_list:
        return gen_texts
