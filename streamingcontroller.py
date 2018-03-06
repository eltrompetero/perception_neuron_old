from soundstreamer import *
import threading
import time
import pygame, sys
import pygame.locals


class SoundStreamingController():
	def __init__:
		self.streamer = SoundStreamer()
		self.pygame = pygame
		self.data = [440,880,220,110,550]

	def init(self):
		self.streamer.init()
		self.pygame.init()

	def update(self):
		self.streamer.update()
		if len(self.streamer.inputbuffer)<10:
			self.streamer.addToStream(np.random.choice(self.data));		
			last = time.time()
		else:
			if time.time() - last > 10:
				self.streamer.inputbuffer = []

		# Key handling
		for event in pygame.event.get():
			if event.type == pygame.locals.KEYDOWN:
				print "key pressed"
				if event.key == pygame.K_p:
					self.streamer.pause()
				if event.key == pygame.K_r:
					self.streamer.resume()
				if event.key == pygame.K_q:
					pygame.quit()
					sys.exit()


# # Soundstreamer
# streamer = SoundStreamer()
# streamer.init()

# streamer.addToStream(10)

# #For Keyboard handling
# pygame.init()


# data = [440,880,220,110,550]

# while True:	
# 	streamer.update()	

# 	# Keep buffer in soundstreamer to length 10
# 	if len(streamer.inputbuffer)<10:
# 		streamer.addToStream(np.random.choice(data));		
# 		last = time.time()
# 	else:
# 		if time.time() - last > 10:
# 			streamer.inputbuffer = []

# 	# Key handling
# 	for event in pygame.event.get():
# 		if event.type == pygame.locals.KEYDOWN:
# 			print "key pressed"
# 			if event.key == pygame.K_p:
# 				streamer.pause()
# 			if event.key == pygame.K_r:
# 				streamer.resume()
# 			if event.key == pygame.K_q:
# 				pygame.quit()
# 				sys.exit()
# 		else:
# 			continue
	
