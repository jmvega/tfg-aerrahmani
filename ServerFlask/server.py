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
import time




app = Flask(__name__)
# app.config.update(
#   SESSION_COOKIE_SECURE=True,
#   SESSION_COOKIE_HTTPONLY=True,
#   SESSION_COOKIE_SAMESITE='Lax'
# )
# csrf=CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
camera=cv2.VideoCapture(0)
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
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/login/2fa/")
@fresh_login_required
def login_2fa():
    secret = pyotp.random_base32()
    server=smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login("correo","contraseña en claro")
    server.sendmail("correo","correo destino",secret)
    return render_template("login_2fa.html", secret=secret)


@app.route("/choose_option/", methods=["GET","POST"])
def get_option():
    if request.method == 'POST':
        if request.form.get("video"):
            return redirect(url_for("video"))
        else:
            return redirect(url_for("login_2fa"))
    else: 
        return redirect(url_for("login_2fa"))


@app.route("/login/2fa/", methods=["POST"])
@fresh_login_required
def login_2fa_form():
  
    secret = request.form.get("secret")
    otp = int(request.form.get("otp"))

    
    if pyotp.TOTP(secret).verify(otp):
      flash("The TOTP 2FA token is valid", "success")
      return render_template("choose_option.html")
    else:
        flash("You have supplied an invalid 2FA token!", "danger")
        return redirect(url_for("login_2fa"))





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
  
  if request.method == 'POST':
    with open('login.txt') as f:
      lines = f.readlines()
    for i in range(len(lines)):
      n=lines[i].split(",")[0]
      user=load_user(n)
      if user.name == n.split(" ")[0] and str(hashpass(user.password)) == n.split(" ")[1].rstrip():
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