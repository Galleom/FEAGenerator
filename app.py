# app.py
from flask import Flask, jsonify
#import tensorflow as tf
import os
#import gc
#import generate


application = Flask(__name__)


@application.route('/')
async def homepage():
    target = os.environ.get('TARGET', 'World')
    return 'Hello {}!\n'.format(target)

if __name__ == '__main__':
    application.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
