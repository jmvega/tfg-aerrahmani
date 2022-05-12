import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from getpass import getpass
import hashlib
salt = os.urandom(32)

filename = "login.txt"
f = open(filename,"a")

add=True
while(add):
    user=input("add new user: ")
    password = getpass("add new user password: ")
    mail=input("add new user email: ")
    password=str.encode(password)
    hashB=hashlib.sha256(password).hexdigest();
    up = user+" "+str(hashB)+" " + mail +"\n"
    f.write(up)
    res = input("Do you want to add a new user? (Y/n): ")
    if res=="n":
        add=False



f.close()

# os.chmod(filename, S_IREAD|S_IRGRP|S_IROTH)