import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt

audio = pyaudio.PyAudio()
l=2048
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=l)

frames = []

start = time.time()
try:
    while True:
        data = stream.read(l)
        frames.append(data)
        
except KeyboardInterrupt:
    pass
end = time.time()
recordtime = end - start

frames2 = b''.join(frames) 
frames = np.frombuffer(frames2, dtype=np.int16)
#numpydata = np.frombuffer(frames2, dtype=np.int16)
#frames = np.array_split(numpydata, l)
#plt.plot(numpydata)
f=[]
f2=[]
f3=[]
f6=[]
amp=[]

Swindow = np.sin(np.arange(l)*np.pi/l)**2
Hwindow = 0.54 + 0.46*(1-np.cos(((2*np.pi)/l)*np.arange(l)))

def freq(f, window=1):
    for i in range(0,len(frames),l):
        amp.append(np.median(frames[i:i+l]))
        power = np.abs(np.fft.rfft(window*frames[i:i+l]))
        f6.append((np.argmax(power*(1-.9*np.linspace(0,1,(l//2+1))))/l)*44100)
        f.append((np.argmax(power)/l)*44100)
   
#filtre médian
#bornes, pour b=3, on prend les 3 valeurs en dessous et au dessus de b, f tableau de fréquences
def filtre_median(f, b=3):
    f1 = np.zeros(len(f))
    for i in range(b,len(f)-(b+1)):
        m=np.median(f[i-b:i+b+1])
        f1[i]=m
    return f1
    
freq(f)
freq(f2, Swindow)

#plt.plot(f2, c = 'grey')
#plt.plot(f, c ='blue')
plt.plot(filtre_median(f,3), c='black')
plt.plot(filtre_median(f6,3), c='red')
#plt.plot(amp[0:(len(amp)//2)], c='green')





#---------------------------------------------test son----------------------------------------------

import pyaudio
import struct
import math

FORMAT = pyaudio.paInt16
CANAUX = 2
TAUX_FREQ = 44100

pa = pyaudio.PyAudio()

def data_freq(frequence: float, temps: float = None):
    # obtient les frames pour une fréquence fixée selon un temps donnée ou un nombre de fréquence donné
    compte_frame = int(TAUX_FREQ * temps)

    remainder_frames = compte_frame % TAUX_FREQ
    ondes = []

    for i in range(compte_frame):
        a = TAUX_FREQ / frequence 
        b = i / a
        c = b * (2 * math.pi)
        d = math.sin(c) * 32767
        e = int(d)
        ondes.append(e)

    for i in range(remainder_frames):
        ondes.append(0)

    nbr_bytes = str(len(ondes))  
    ondes = struct.pack(nbr_bytes + 'h', *ondes)

    return ondes   

def play(frequence: float, temps: float):
    """
    joue la fréquence selon un temps donné
    """
    frames3 = data_freq(frequence, temps)
    stream = pa.open(format=FORMAT, channels=CANAUX, rate=TAUX_FREQ, output=True)
    stream.write(frames3)
    stream.stop_stream()
    stream.close()

N=[]
fmed = filtre_median(f,4)
for i in range(1,len(fmed)-1):
    if fmed[i]>16.34:
        if fmed[i-1]==fmed[i]==fmed[i+1]:
            N[len(N)-1][1] += 1 
        elif fmed[i]==fmed[i+1]:
            N.append([fmed[i],2])

print(N,fmed)

S = 0

for i in N:
    if i[1]<=1:
        N.remove(i)
    else:
        S += i[1]
        
print(S)

for i in N:
    #play(i[0], (recordtime/S)*i[1]+0.5)
        
'''
fmed = filtre_median(f,3)
for i in range(len(fmed)):
    print(fmed[i])
    if fmed[i] > 50 and amp[0:(len(amp)//2)]>50 and round(fmed[i])!=round(fmed[i-1]) and amp:
        play(fmed[i], 1)
        time.sleep(1)'''
        
        
#-----------------------------------------------database------------------------------------------------

import sqlite3

conn = sqlite3.connect('rec.db')
cursor = conn.cusor()
cursor.execute("""CREATE TABLE IF NOT EXISTS rec_db (
    track_name string,
    duration int,
    track list,
);  """)

cursor.execute("INSERT INTO rec_db (track_name, duration, track) VALUES (?,?,?)" (None,recordtime,N,))
