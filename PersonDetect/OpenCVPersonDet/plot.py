import numpy as np
import matplotlib.pyplot as plt
import sys

def usage():
    print("Usage: python3 plot_results.py [-h/--help] [FILE]")
    exit()

def help():
    print("plot_results.py es un programa que crea gráficas a partir de arrays guardados en archivos \".npy\".")
    print("Para usarlo, debes ejecutar el siguiente comando cambiando \"FILE\" por el archivo o archivos que se quieran representar")
    print("\n\
        python3 plot_results [FILE]\n")
    exit()


files = []
data = []

for i in range(len(sys.argv)-1):
    # Todos los parametros tienen que ser archivos tipo .npy or el argumento "-h"
    if(sys.argv[i+1] == "-h" or sys.argv[i+1] == "--help"):
        help()
    if(sys.argv[i+1][sys.argv[i+1].find(".")+1:] != "npy"):
        usage()
    files.append(sys.argv[i+1])

fig, ax = plt.subplots()  # Create a figure and an axes.

for i in range(len(files)):
    data.append(np.load(files[i]))

    pos = data[i][0:len(data[i]),0]
    speed = data[i][0:len(data[i]),1]


    ax.plot(pos, speed, label=files[i][:files[i].find("D")-1])

ax.set_xlabel('tiempo')  # Add an x-label to the axes.
ax.set_ylabel('frames')  # Add a y-label to the axes.
ax.set_title("Práctica 1")  # Add a title to the axes.

ax.legend()  # Add a legend.
plt.show()