import cv2
from multiprocessing import Process,Pipe
from detect import run
import socket


if __name__=='__main__':
    ss=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    ss.bind(("",8050))
    ss.listen(1)

    cli,addr=ss.accept()
    while True:
        rec=cli.recv(320*240)

        print(rec)

    cli.close()
    ss.close()
    