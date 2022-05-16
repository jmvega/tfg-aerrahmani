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
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
import sys
import time
import cv2
from maskdetector import MaskDetector
import numpy as np
import sqlite3 as sq




cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
db = 'secure.db'
mail_to=""

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


@app.route("/choose_option/", methods=["GET","POST"])
def get_option():
    if request.method == 'POST':
        if request.form.get("video"):
            return redirect(url_for("video"))
        else:
            return redirect(url_for("gallery"))
    else: 
        return redirect(url_for("login_2fa"))


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
    count=0
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

            if class_name=="no_mask":
              count+=1
              cv2.imwrite("images/frame%d.jpg"%count,image)
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
    return render_template("choose_option.html")
    
    if pyotp.TOTP(secret).verify(otp):
        flash("The TOTP 2FA token is valid", "success")
        return redirect(url_for("video"))
    else:
        flash("You have supplied an invalid 2FA token!", "danger")
        return redirect(url_for("login_2fa"))


def search_user_data(name):
    conn = sq.connect(db)
    c = conn.cursor()
    c.execute('SELECT * FROM data WHERE user="%s"'%name)
    data=c.fetchall()
    conn.close()
    if len(data)==1:
        return data
    if len(data) == 0:
        return None

@login_manager.user_loader
def load_user(n):
  global users
  form_user = request.form.get("username")

  form_pass = request.form.get("password")
  user=User(1,form_user,form_pass)
  users.append(user)
  return user

@app.route("/login", methods=['GET', 'POST'])
def login():
  global mail_to
  if request.method == 'POST':
      user=load_user(1)
      data=search_user_data(user.name)
      if data != None:
        if user.name == data[0][1] and str(hashpass(user.password)) == data[0][2]:
          mail_to=data[0][0]
          login_user(user,remember=False)
          return redirect(url_for("login_2fa"))
  flash("Invalid credentials. Please try again.")
  return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join('images/')
    if not os.path.isdir(target):
            os.mkdir(target)
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route("/gallery")
@fresh_login_required
def gallery():
  image_names = os.listdir("images")
  return render_template("gallery.html",image_names=image_names)

@app.route("/video")
@fresh_login_required
def video():
  return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
  
  app.config["SECRET_KEY"] = os.urandom(16).hex()
  app.run(debug=False,host="0.0.0.0")