from soundstreamer import *
import threading
import time
import pygame, sys
import pygame.locals

# Soundstreamer
ss = SoundStreamer()
ss.init()

ss.addToStream(10)

#For Keyboard handling
pygame.init()

data = [440,880,220,110,550]

while True:	
	ss.update()	

	# Keep buffer in soundstreamer to length 10
	if len(ss.inputbuffer)<10:
		ss.addToStream(np.random.choice(data));		
		last = time.time()
	else:
		if time.time() - last > 10:
			ss.inputbuffer = []

	# Key handling
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
	
