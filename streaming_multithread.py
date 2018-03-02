from soundstreamer import *
import threading
import time

ss = SoundStreamer()

ss.start()
startThread = threading.Thread(target = ss.update)
endThread = threading.Thread(target = ss.pause)


startThread.start()
print "endThread start"
time.sleep(10)
endThread.start()