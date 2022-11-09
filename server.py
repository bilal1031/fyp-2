from flask import Flask, render_template, request
import eventlet
import socketio
import eventlet.wsgi

sio = socketio.Server()#async_mode=async_mode)
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

dict1={}
i=0
@app.route('/')
def index():
	return render_template('file.html')

@sio.event()
def pingpong(sid):
	print("//////////////////////////")
	sio.emit("send_data", room=sid)

@sio.event
def connect(sid, data):	
	print("[INFO] Connect to the server")
	pingpong(sid)

@sio.event
def send(sid, data):
	global i
	if sid not in dict1:
		i+=1
		dict1[sid]=i
	key=dict1[sid]
	print("Reached here")
	sio.emit('response',{'key':key, 'data':data})
	pingpong(sid)

@sio.event
def disconnect(sid):
	print("[INFO] disconnected from the server")

if __name__ == '__main__':
	eventlet.wsgi.server(eventlet.listen(('localhost',5000)), app)


# from flask import Flask, request, Response,render_template,make_response
# import jsonpickle
# import numpy as np
# import cv2
# import requests
# import base64
# import io
# from PIL import Image
# import base64
# # Initialize the Flask application
# app = Flask(__name__)

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# #route http posts to this method
# gimg = 0
# @app.route('/', methods=['POST'])
# def test():
#     r = request
#     # convert string of image data to uint8
#     nparr = np.fromstring(r.data, np.uint8)
#     # decode image
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     print(img)
#     # do some fancy processing here....

#     # build a response dict to send back to client
#     response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
#                 }
#     # encode response using jsonpickle
#     response_pickled = jsonpickle.encode(response)

    
#     return Response(response=response_pickled, status=200, mimetype="application/json")


# def render_frame(arr: np.ndarray):
#     mem_bytes = io.BytesIO()
#     img = Image.fromarray(arr)
#     img.save(mem_bytes, 'JPEG')
#     mem_bytes.seek(0)
#     img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
#     mime = "image/jpeg"
#     uri = "data:%s;base64,%s"%(mime, img_base64)
#     return render_template("index.html", image=uri)

# def getImg():
#     while True:
#         yield (np.random.random((300,400, 3)) * 255).astype("uint8")

# @app.route("/", methods=['GET'])
# def main():
#     return render_frame(getImg())


# # start flask app
# # if __name__ == '__main__':
# app.run(host="0.0.0.0", port=5000)