# app.py
from flask import Flask, jsonify
import tensorflow as tf
import os
import gc


app = Flask(__name__)

import generate

sess = generate.start_tf_sess(threads=1)
generate.load_gpt2(sess, model_name="text_model")

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}
generate_count = 0

@app.route('/')
def homepage():
    global generate_count
    global sess
    text = generate.generate(sess,
                      length=1024,
                      model_name="text_model",
                      include_prefix=False,
                      return_as_list=True
                         )[0]
    generate_count += 1
    if generate_count == 8:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = generate.start_tf_sess(threads=1)
        generate.load_gpt2(sess)
        generate_count = 0
        
    gc.collect()
    return jsonify({'text': text},
                         headers=response_header)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
