import numpy as np
import random
from collections import deque


class SnakeGame:
	def __init__(self, grid_size):
		assert grid_size >= 5
		self.length =2
		self.grid_size = grid_size
		self.action_space = ["forward","left","right"]
		self.eat = 10 
		self.is_wall =[]
		self.WallCollisionReward = -10
		self.BodyCollisionReward = -10

		


	
		 
		# specify this better


	def reset(self):
		self.face_direction = 'right'
		if self.grid_size % 2 == 0:
			self.h_r = int((self.grid_size/2)-1)
			self.h_c = int((self.grid_size/2)-1)
		else:
			self.h_r = int((self.grid_size-1) /2)
			self.h_c = int((self.grid_size-1) /2)
		
		# initializing snake, leftmost element is the head
		self.snake_indices =deque()
		for ele in range(self.length):
			self.snake_indices.append((self.h_r, self.h_c-self.length+1+ele))
		self.food = self.new_food_index()
		self.reward = 0
		self.done = False
		self.info = None 
		return {'state':self.render(), 'reward':self.reward, 'info' : self.info, 'done':self.done,'face_direction': self.face_direction}

	def render(self):
		self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
		for ele in range(len(self.snake_indices)):
			if ele == len(self.snake_indices)-1: # head
				self.state[self.snake_indices[ele]] = 1
			else:	
				self.state[self.snake_indices[ele]] = 2
		for ele in range(len(self.snake_indices)-1):
			if self.snake_indices[-1] == self.snake_indices[ele]:
		# if head index is the same as any body's index 
				self.info = 'body_collision'
				self.done = True
				self.reward += self.BodyCollisionReward


		self.state[self.food] = 3
		return self.state


	def new_food_index(self):
		r= random.randint(0,self.grid_size-1)
		c= random.randint(0,self.grid_size-1)
		while (r,c) in set(self.snake_indices):
			r= random.randint(0,self.grid_size-1)
			c= random.randint(0,self.grid_size-1)
		return (r,c)

	def final_step(self, final_action):
		old_length = len(self.snake_indices)
		old_state = self.render()
		if final_action == 'right':
			if old_state[self.snake_indices[-1][0], self.snake_indices[-1][1]+1] == 3:
				self.reward += self.eat
				self.snake_indices.appendleft((0,0))
				self.food = self.new_food_index()
			x=0
			y=1
		if final_action == 'left':
			if old_state[self.snake_indices[-1][0], self.snake_indices[-1][1]-1] == 3:
				self.reward += self.eat
				self.snake_indices.appendleft((0,0))
				self.food = self.new_food_index()
			x= 0
			y= -1
		if final_action == 'up':
			if old_state[self.snake_indices[-1][0]-1, self.snake_indices[-1][1]] == 3:
				self.reward += self.eat
				self.snake_indices.appendleft((0,0))
				self.food = self.new_food_index()
			x= -1
			y= 0
		if final_action == 'down':
			if old_state[self.snake_indices[-1][0]+1, self.snake_indices[-1][1]] == 3:
				self.reward += self.eat
				self.snake_indices.appendleft((0,0))
				self.food = self.new_food_index()
			x= 1
			y= 0
		self.face_direction = final_action
		snake_indices = self.deque_update(x,y)
		self.is_wall = self.iswall()
		if len(self.snake_indices) > old_length: 
			food_index = self.new_food_index()
		return self.render()

	def step(self, action):
		key = action
		if self.face_direction == 'up':
			dic = {'forward':'up', 'left':'left', 'right':'right'}
		if self.face_direction == 'down':
			dic = {'forward': 'down', 'left':'right', 'right':'left'}
		if self.face_direction == 'left':
			dic = {'forward': 'left', 'left':'down', 'right':'up'}
		if self.face_direction == 'right':
			dic = {'forward':'right', 'left':'up', 'right':'down'}

		if action not in set(self.is_wall):
			return {'state':self.final_step(dic[key]), ",reward":self.reward, "info":self.info,
				"done":self.done, 'face_direction':self.face_direction}
		else:
			# wall collision
			self.reward += self.WallCollisionReward
			self.done = True
			self.info = 'wall_collision'
			return {'state':self.render(), "reward":self.reward,'info':self.info,
				"done":self.done, 'face_direction':dic[key]}



	def deque_update(self, x,y):
		(a,b) = self.snake_indices[-1]
		if -1 < (a+x) <self.grid_size and -1<  (b+y) < self.grid_size:
			# if (a+x, b+y) in set(self.snake_indices):
				# game_over = True
			self.snake_indices.append((self.snake_indices[-1][0]+x, self.snake_indices[-1][1]+y))
			self.snake_indices.popleft()
			return self.snake_indices
		else:
			# WALL COLLISION happens, taken care in 

			return self.snake_indices
		#reset the game

	def iswall(self):	
		wall = []
		if self.snake_indices[-1][0] == 0:
			if self.face_direction == 'up':
				wall.append('forward')
			if self.face_direction == 'left':
				wall.append('right')
			if self.face_direction == 'right':
				wall.append('left')
		if self.snake_indices[-1][0] == self.grid_size-1:
			if self.face_direction == 'down':
				wall.append('forward')
			if self.face_direction == 'left':
				wall.append('left')
			if self.face_direction == 'right':
				wall.append('right')
		if self.snake_indices[-1][1] == 0:
			if self.face_direction == 'up':
				wall.append('left')
			if self.face_direction == 'left':
				wall.append('forward')
			if self.face_direction == 'down':
				wall.append('right')
		if self.snake_indices[-1][1] == self.grid_size-1:
			if self.face_direction == 'up':
				wall.append('right')
			if self.face_direction == 'down':
				wall.append('left')
			if self.face_direction == 'right':
				wall.append('forward')
		return wall

# FIN
