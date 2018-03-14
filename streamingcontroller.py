from soundstreamer import *
import threading
import time
import pygame, sys
import pygame.locals


class SoundStreamingController():
	def __init__(self):
		self.streamer = SoundStreamer()
		self.pygame = pygame

	def init(self):
		self.streamer.init()
		self.pygame.init()

	def feedData(self, data):
		self.streamer.addToStream(data)

	def close(self):
		self.streamer.close()
		self.pygame.quit()

	def getStreamer(self):
		return self.streamer

	def update(self):
		self.streamer.update()
		# Key handling
		# for event in self.pygame.event.get():
		# 	if event.type == self.pygame.locals.KEYDOWN:
		# 		print "key pressed"
		# 		if event.key == pygame.K_p:					
		# 			self.streamer.pause()
		# 			return False
		# 		if event.key == pygame.K_r:
		# 			self.streamer.resume()
		# 			return True
		# 		if event.key == pygame.K_q:
		# 			self.pygame.quit()
		# 			return False

		return True
