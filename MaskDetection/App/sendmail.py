from flask import Flask
from flask_mail import Mail, Message
import smtplib

server=smtplib.SMTP("smtp.gmail.com",587)
server.starttls()
server.login("correo@gmail.com","contraseña en claro")
server.sendmail("correo@gmail.com","correodestinatario@gmail.com","secret")