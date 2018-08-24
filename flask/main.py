# When running locally install requirements using
# pip3 install -t lib -r requirements.txt
# FLASK_APP=main.py FLASK_DEBUG=1 python -m flask run

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

@app.route('/health_scores')
def health_scores():
    return render_template('health_scores.html')

# {'loss': 0.10061083, 'global_step': 20000, 'accuracy': 0.9701}
@app.route('/draw', methods=['GET', 'POST'])
def draw():
    if request.method == 'POST':
        print("waiting on post")
        dataURI = request.data.decode('UTF-8')
        image_data = re.sub('^data:image/png;base64,', '', dataURI)
        im = Image.open(BytesIO(base64.b64decode(image_data)))
  
        # If a 1 is bounding boxed, it will explode to a fat line.
        # Need a better resize tool. maybe just center it on a blank canvas?
        # Maybe Check the aspect ratio before cropping? 
        b_box = im.getbbox()
        if b_box is None:
            return jsonify(prediction="Draw something first")
        # Expand bounding box, in order to center the image a bit more.
        expansion_coef = (-5,-5,5,5)
        b_box2 = [sum(x) for x in zip(b_box,expansion_coef )]
        im = im.crop(b_box2)

        im = im.resize((28,28), Image.ANTIALIAS)

        # For visualizing the crop and rescale.
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        data64 = base64.b64encode(buffered.getvalue())
        img_str = u'data:img/png;base64,'+ data64.decode('utf-8')

        pixels = list(im.getdata())
        # Turn pixels to B&W if they are over $threshold value.
        threshold = 150
        # Color value seems is stored in the transparency channel.
        #b_w = list(map(lambda rgba: 1 if sum(rgba) >= threshold else 0, pixels))
        b_w = list(map(lambda rgba: rgba[3]/255.0, pixels))
        data = np.array(b_w)
        data.shape = (28,28)
        data = data.tolist()
        # Prepare to send to ML Instance
        req = {"instances": [{"x": data}]}
        prediction = mnist_prediction(service, req)
        predicted_class = prediction['predictions'][0]['classes']
        top_3 = sorted(enumerate(prediction['predictions'][0]['probabilities']),key=lambda x: x[1], reverse=True)[0:3]
        print(top_3[0:3])
        return jsonify(prediction=top_3, img_uri=img_str)
    return render_template('digits.html')
    # Potentially add another POST where we add the user corrected result to an
    # "ON DEMAND" First Generation MYSQL instance. Or just regular DB hosted in a bucket?
    # Also make this not RPC to a ML-Engine, the spin up time is too long.

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
