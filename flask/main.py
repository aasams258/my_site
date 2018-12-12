# When running locally install requirements using
# pip3 install -t lib -r requirements.txt
# then run with
# FLASK_APP=main.py FLASK_DEBUG=1 python -m flask run

import base64, re
import cloud_utils
import numpy as np
from flask import Flask, jsonify, request, render_template
from io import BytesIO
from PIL import Image
from googleapiclient import discovery


app = Flask(__name__)

service = discovery.build('ml', 'v1')

@app.route('/')
def index():
    return render_template('homepage.html')

"""
General Landing Page for projects.
"""
@app.route('/projects')
def projects():
    return render_template('projects.html')

"""
LA county health scores project.
"""
@app.route('/health_scores')
def health_scores():
    return render_template('health_scores.html')

"""
Draw a digit and predict with MNIST layers project.
If RPC to ML-Engine spin up time is too long, host model locally.
"""
@app.route('/digits', methods=['GET', 'POST'])
def digits():
    if request.method == 'POST':
        dataURI = request.data.decode('UTF-8')
        image_data = re.sub('^data:image/png;base64,', '', dataURI)
        drawn_input = Image.open(BytesIO(base64.b64decode(image_data)))

        bounding_box = drawn_input.getbbox()
        if bounding_box is None:
            return jsonify(error="Draw something first")

        # Expand bounding box, in order to center the image a bit more.
        expansion_coef = (-20, -20, 20, 20)
        enlarged_boundary = [sum(x) for x in zip(bounding_box, expansion_coef)]
        drawn_input = drawn_input.crop(enlarged_boundary)
        drawn_input = drawn_input.resize((28, 28), Image.ANTIALIAS)

        # For visualizing the crop and rescale.
        buffered = BytesIO()
        drawn_input.save(buffered, format="PNG")
        data64 = base64.b64encode(buffered.getvalue())
        img_str = u'data:img/png;base64,'+ data64.decode('utf-8')

        pixels = list(drawn_input.getdata())
        # For the PNG the color is in the alpha channel. Normalize to [0,1].
        b_w = list(map(lambda rgba: rgba[3]/255.0, pixels))
        data = np.array(b_w)
        data.shape = (28, 28)
        data = data.tolist()
        # Prepare to send to ML Instance.
        req = {"instances": [{"x": data}]}
        try:
            prediction = cloud_utils.mnist_prediction(service, req)
            top_3 = sorted(enumerate(prediction['predictions'][0]['probabilities']),
                           key=lambda x: x[1],
                           reverse=True)[0:3]
            return jsonify(prediction=top_3, img_uri=img_str)
        except:
            return jsonify(error="Error in the ML Instance, Please try again.")
    return render_template('digits.html')
