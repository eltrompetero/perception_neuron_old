from soundstreamer import *
import threading
import time
import pygame, sys
import pygame.locals

ss = SoundStreamer()

ss.init()
# startThread = threading.Thread(target = ss.update)
# addThread = threading.Thread(target = ss.addToStream, args = (10,))
# endThread = threading.Thread(target = ss.pause)
ss.addToStream(10);
pygame.init()

data = [440,880,220,110,550]

while True:	
	ss.update()	
	if len(ss.inputbuffer)<10:
		ss.addToStream(np.random.choice(data));		
		last = time.time()
	else:
		if time.time() - last > 10:
			ss.inputbuffer = []
	for event in pygame.event.get():
		if event.type == pygame.locals.KEYDOWN:
			print "key pressed"
			if event.key == pygame.K_p:
				ss.pause()
			if event.key == pygame.K_r:
				ss.resume()
			if event.key == pygame.K_q:
				pygame.quit()
				sys.exit()
		else:
			continue
	
