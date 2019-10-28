# app.py
from flask import Flask, request, jsonify
import os
app = Flask(__name__)

import generate
def init():
    global graph
    graph = generate.load_graph('.', 'text_model')

@app.route('/getmsg/', methods=['GET'])
def respond():
    return "<h1>Welcome to our server !!</h1>"

@app.route('/result', methods=['GET'])
def get_something():
    return jsonify({
        "Message": generate.predict(graph),
        # Add this option to distinct the POST request
        "METHOD" : "GET"
    })

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

# A welcome message to test our server
@app.route('/')
def index():
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
    app.run(threaded=True, port=(int(os.environ.get("PORT", 5000))))