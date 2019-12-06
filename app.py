# app.py
from flask import Flask, request, jsonify
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

@app.route('/', methods=['GET'])
def index():
    global generate_count
    global sess
    params = request.query_params
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

@app.route('/results/', methods=['POST'])
def post_something():
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": f"Welcome {param} to our awesome platform!!",
            # Add this option to distinct the POST request
            "METHOD" : "POST"
        })
    else:
        return jsonify({
            "ERROR": "no name found, please send a name."
        })

@app.route('/getmsg/', methods=['GET'])
def respond():
# A welcome message to test our server
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"got name {name}")

    response = {}

    # Check if user sent a name at all
    if not name:
        response["ERROR"] = "no name found, please send a name."
    # Check if the user entered a number not a name
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    # Now the user entered a valid name
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"

    # Return the response in json format
    return jsonify(response)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, host='0.0.0.0', port=(int(os.environ.get("PORT", 8080))))
