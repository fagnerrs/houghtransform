from flask import render_template, request, send_file
from app import app
import os

from app.models.Parabola import Parabola

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
  return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():

  for upload in request.files.getlist("file"):
    filename = upload.filename

    # This is to verify files are supported
    ext = os.path.splitext(filename)[1]
    destination = "/".join([os.path.join(APP_ROOT, 'images'), filename])

    upload.save(destination)

  numInliers = int(request.form['num-inliers'])
  numThreshold = float(request.form['num-threshold'])
  numIteractions = int(request.form['num-iteractions'])
  numParabolaPoints = int(request.form['num-parabola-points'])

  parabola = Parabola()

  parabola.find(destination, numThreshold, numInliers, numParabolaPoints, numIteractions)

  # return send_from_directory("images", filename, as_attachment=True)
  return send_file(destination, mimetype='image/gif')