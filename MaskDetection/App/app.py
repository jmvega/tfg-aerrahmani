from urllib import response
from flask import *
import os
import sys
import hashlib
import cv2
import pyotp
from flask_bootstrap import Bootstrap
import smtplib
from flask_login import *
# from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
import sys
import time
import cv2
from maskdetector import MaskDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)



app = Flask(__name__)
# app.config.update(
#   SESSION_COOKIE_SECURE=True,
#   SESSION_COOKIE_HTTPONLY=True,
#   SESSION_COOKIE_SAMESITE='Lax'
# )
# csrf=CSRFProtect(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
Bootstrap(app)


class User(UserMixin):

    def __init__(self, id, name, password):
        self.id = id
        self.name = name
        self.password = password
        # self.email = email

users=[]

@app.route("/")
def index():
  return render_template('index.html')


def hashpass(passw):
  password=str.encode(passw)
  hashB=hashlib.sha256(password).hexdigest()
  
  return hashB


def generate_frames():
    threads=4

    counter, fps = 0, 0
    start_time = time.time()
    


    row_size = 20
    left_margin = 24
    text_color = (0, 0, 255)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    init_time=time.time()

    
    detector = MaskDetector(model_path="maskdef.tflite")

    NUM_DATA=2
    measures = np.zeros([1,NUM_DATA])
    first_time=True
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1

        image=cv2.resize(image,(320,240))


        detections = detector.detect(image)

        for detection in detections:

            start_point = detection.bounding_box.left, detection.bounding_box.top
            end_point = detection.bounding_box.right, detection.bounding_box.bottom
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)


            category = detection.categories[0]
            class_name = category.label
            probability = round(category.score, 2)
            result_text = class_name + ' (' + str(probability) + ')'
            text_location = (_MARGIN + detection.bounding_box.left,
                            _MARGIN + _ROW_SIZE + detection.bounding_box.top)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
        # if counter % fps_avg_frame_count == 0:
        #     end_time = time.time()
        #     fps = fps_avg_frame_count / (end_time - start_time)
        #     start_time = time.time()

        # fps_text = 'FPS = {:.1f}'.format(fps)

        # text_location = (left_margin, row_size)
        # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
        #             font_size, text_color, font_thickness)
        ret,buffer=cv2.imencode('.jpg',image)
        image=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')



@app.route("/login/2fa/")
@fresh_login_required
def login_2fa():
    secret = pyotp.random_base32()
    server=smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login("sendmessagefromflaak@gmail.com","123456fl*")
    server.sendmail("sendmessagefromflaak@gmail.com","arraklk924@gmail.com",secret)
    return render_template("login_2fa.html", secret=secret)



@app.route("/login/2fa/", methods=["POST"])
@fresh_login_required
def login_2fa_form():
  
    secret = request.form.get("secret")
    otp = int(request.form.get("otp"))

    
    if pyotp.TOTP(secret).verify(otp):
        flash("The TOTP 2FA token is valid", "success")
        return redirect(url_for("video"))
    else:
        flash("You have supplied an invalid 2FA token!", "danger")
        return redirect(url_for("login_2fa"))


@login_manager.user_loader
def load_user(n):
  form_user = request.form.get("username")
  form_pass = request.form.get("password")
  user=User(1,form_user,form_pass)
  return user

@app.route("/login", methods=['GET', 'POST'])
def login():
  
  if request.method == 'POST':
    with open('login.txt') as f:
      lines = f.readlines()
    for i in range(len(lines)):
      n=lines[i].split(",")[0]
      user=load_user(n)
      if user.name == n.split(" ")[0] and str(hashpass(user.password)) == n.split(" ")[1].rstrip():
        # app.config["MAIL_USERNAME"] = n.split(" ")[2].rstrip()
        
        login_user(user,remember=True)
        return redirect(url_for("login_2fa"))
    flash("Invalid credentials. Please try again.")
  return redirect(url_for("index"))



@app.route("/video")
@fresh_login_required
def video():
  return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
  
  app.config["SECRET_KEY"] = os.urandom(16).hex()
  app.run(debug=False,host="0.0.0.0")