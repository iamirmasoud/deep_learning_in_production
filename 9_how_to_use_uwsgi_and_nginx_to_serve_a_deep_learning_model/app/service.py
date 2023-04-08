import os
import traceback
import sys
from flask import Flask, jsonify, request

sys.path.append("{}".format(os.getcwd()))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'executor')))
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "executor"

from executor.unet_inferrer import UnetInferrer


app = Flask(__name__)

APP_ROOT = os.getenv("APP_ROOT", "/infer")
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv("PORT_NUMBER", 8089))

u_net = UnetInferrer()


@app.route(APP_ROOT, methods=["POST"])
def infer():
    data = request.json
    image = data["image"]
    return u_net.infer(image)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == "__main__":
    app.run(host=HOST, port=PORT_NUMBER)
