# When running locally install requirements using
# pip3 install -t lib -r requirements.txt

# [START app]
import logging
import base64, re

from flask import Flask, jsonify, request, render_template
import sys
from io import BytesIO
from PIL import Image

#Repackage these 2.
from cloud_utils import mnist_prediction
import numpy as np
from googleapiclient import discovery

app = Flask(__name__)

service = discovery.build('ml', 'v1')

@app.route('/')
def index():
    return render_template('homepage.html')

# {'loss': 0.10061083, 'global_step': 20000, 'accuracy': 0.9701}
@app.route('/draw', methods=['GET', 'POST'])
def draw():
    if request.method == 'POST':
        print("waiting on post")
        dataURI = request.data.decode('UTF-8')
        image_data = re.sub('^data:image/png;base64,', '', dataURI)
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        '''
        # If a 1 is bounding boxed, it will explode to a fat line.
        # Need a better resize tool. maybe just center it on a blank canvas?
        b_box = im.getbbox()
        if b_box is None:
            return jsonify(prediction="Draw something first")
        im = im.crop(b_box)
        im = im.resize((28,28), Image.ANTIALIAS)
        '''
        '''
        # For outputing a data URI
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        print(img_str)
        '''
        im.thumbnail((28,28), Image.ANTIALIAS)
        pixels = list(im.getdata())
        # Turn pixels to B&W if they are over $threshold value.
        threshold = 150
        # Color value seems to be stored in the transparency channel.
        # But will summate over all just incase.
        b_w = list(map(lambda rgba: 1 if sum(rgba) >= threshold else 0, pixels))
        data = np.array(b_w)
        data.shape = (28,28)
        data = data.tolist()
        # Prepare to send to ML Instance
        req = {"instances": [{"x": data}]}
        prediction = mnist_prediction(service, req)
        predicted_class = prediction['predictions'][0]['classes']
        return jsonify(prediction=predicted_class)
    return render_template('digits.html')
    # Potentially add another POST where we add the user corrected result to an
    # "ON DEMAND" First Generation MYSQL instance. Or just regular DB hosted in a bucket?
    # Also make this not RPC to a ML-Engine, the spin up time is too long.

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
# [END app]