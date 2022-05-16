import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from getpass import getpass
import hashlib
import sqlite3 as sq
 
salt = os.urandom(32)
db = 'secure.db'
 
 
def connect():
   

    conn = sq.connect(db)

    c = conn.cursor()

    c.execute("""
                 CREATE TABLE IF NOT EXISTS data (
                     site text,
                     user text,
                     password text primary key
 
                 )             
    """)
     

    conn.commit()
    conn.close()
 
 
def enter(site, user, pas):
    conn = sq.connect(db)
    c = conn.cursor()
    c.execute("INSERT INTO data VALUES(?,?,?)", (email, user, pas))
    conn.commit()
    conn.close()
 
 
def show():
    conn = sq.connect(db)
    c = conn.cursor()
    c.execute("SELECT * FROM data")

    i = c.fetchall()
    conn.commit()
    conn.close()
    return i
 
 
def Del(password):
    conn = sq.connect(db)
    c = conn.cursor()
    c.execute("DELETE FROM data WHERE password=(?)", (password,))
    conn.commit()
    conn.close()
 
 
def check():

    if len(show()) == 0:
        return False
    else:
        return True



add=True
connect()
while(add):
    res = input("Do you want to add a new user? (Y/n): ")
    if res=="n":
        break
    user=input("add new user: ")
    password = getpass("add new user password: ")
    mail=input("add new user email: ")
    password=str.encode(password)
    hashB=hashlib.sha256(password).hexdigest();
    enter(mail,user,hashB)

    



 
