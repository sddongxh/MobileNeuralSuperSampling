import os,sys
import cv2


mode = "Q"
path, savename= "", ""
if mode == "Q":
  path=r".\checkpoint\results\Q"
  savename = r".\Videos\Q.avi"
elif mode== "B":
  path=r".\checkpoint\results\B"
  savename = r".\Videos\B.avi"
elif mode== "P":
  path=r".\checkpoint\results\P"
  savename = r".\Videos\P.avi"
elif mode== "GT":
  path=r".\checkpoint\results\GT"
  savename = r".\Videos\GT.avi"
elif mode== "LR":
  path=r".\Data\1080p\LR"
  savename = r".\Videos\LR.avi"

pic=os.listdir(path)
pic.sort(key=lambda x: int(x[:-4]))
w = 1920
h = 1080
f = 2

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(savename, fourcc, 60, (w, h))
for p in pic:
  print(p)
  im=cv2.imread(path+'/'+p)
  im = cv2.resize(im, (w,h), interpolation=cv2.INTER_CUBIC)
  for i in range(f):
    out.write(im)
out.release()
