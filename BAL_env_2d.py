
import numpy as np
np.set_printoptions(suppress=True)
import copy
import warnings
import collections
import time
from gym import spaces, core
import gym
import sys
from gym.spaces import Box
from gym.utils import seeding
import math
import random as rand

class bal_2denv(gym.Env):
    def __init__(self):
        # action and observation space
        self._action_low = -np.array([1,1])
        self._action_high = -self._action_low
        self._action_space = Box(low=self._action_low, high=self._action_high)
        self._observation_low = -np.ones((1,4))
        self._observation_high = -self._observation_low
        self._observation_space = Box(low=self._observation_low, high=self._observation_high)
        self.cam_init_pos_space = 10
        self.cambase_init_ori_space = [20,20,360]
        self.cam12_init_ori_space = [10,10,60]

    def reset(self):
        cam_base_position = (self.cam_init_pos_space*np.random.rand(1,3) - 5*np.ones((1,3)))
        cam_base_orientation = self.get_rot_mat(self.cambase_init_ori_space,random='y')
        self.bottom = np.array([[0,0,0,1]])
        self.CAM_base = np.concatenate((np.concatenate((cam_base_orientation,cam_base_position.T),axis=1),self.bottom),axis=0)
        cam1_pos = np.matmul(self.CAM_base,np.array([[rand.uniform(-3,0)-5,0,0,1]]).T)[0:-1,:]
        cam2_pos = np.matmul(self.CAM_base,np.array([[rand.uniform(0,3)+5,0,0,1]]).T)[0:-1,:]
        cam1_ori = np.matmul(cam_base_orientation,self.get_rot_mat(self.cam12_init_ori_space,random='y'))
        cam2_ori = np.matmul(cam_base_orientation,self.get_rot_mat(self.cam12_init_ori_space,random='y'))

        self.CAM_1 = np.concatenate((np.concatenate((cam1_ori,cam1_pos),axis=1),self.bottom),axis=0)
        self.CAM_2 = np.concatenate((np.concatenate((cam2_ori,cam2_pos),axis=1),self.bottom),axis=0)

        self.CAM_base[0:3,3] = self.CAM_base[0:3,3]/100
        self.CAM_1[0:3,3] = self.CAM_1[0:3,3]/100
        self.CAM_2[0:3,3] = self.CAM_2[0:3,3]/100

        self.CAM1_to_CBase = np.matmul(np.linalg.inv(self.CAM_base),self.CAM_1)
        self.CAM2_to_CBase = np.matmul(np.linalg.inv(self.CAM_base),self.CAM_2)

        self.bolt_3d_pos, self.bolt_PCs = self.create_bolt()
        self.target_3d_pos = self.create_target()
        
        self.bolt1_in_left_cam = self.convert_from_base_2_img(self.CAM_1,self.bolt_3d_pos[:,0])
        self.bolt2_in_left_cam = self.convert_from_base_2_img(self.CAM_1,self.bolt_3d_pos[:,1])
        self.bolt1_in_right_cam = self.convert_from_base_2_img(self.CAM_2,self.bolt_3d_pos[:,0])
        self.bolt2_in_right_cam = self.convert_from_base_2_img(self.CAM_2,self.bolt_3d_pos[:,1])
        self.target_in_left_cam = self.convert_from_base_2_img(self.CAM_1,self.target_3d_pos)
        self.target_in_right_cam = self.convert_from_base_2_img(self.CAM_2,self.target_3d_pos)

        self.bolt_vec_left = self.bolt2_in_left_cam - self.bolt1_in_left_cam
        self.bolt_vec_right = self.bolt2_in_right_cam - self.bolt1_in_right_cam
        self.track_vec_left = self.target_in_left_cam - self.bolt2_in_left_cam
        self.track_vec_right = self.target_in_right_cam - self.bolt2_in_right_cam
        
        self.bolt_vec_left = self.bolt_vec_left/np.linalg.norm(self.bolt_vec_left,2)
        self.bolt_vec_right = self.bolt_vec_right/np.linalg.norm(self.bolt_vec_right,2)
        
        self.coord_mat_left, self.coord_mat_right = self.get_coord_mat()

        state = self.observation()
     
        # assert self.observation_space.contains(state) == True
        return state


    def step(self,action):
        action = action*0.05
        self.CAM_base[0:2,3] = self.CAM_base[0:2,3] + action

        self.CAM_1 = np.matmul(self.CAM_base,self.CAM1_to_CBase)
        self.CAM_2 = np.matmul(self.CAM_base,self.CAM2_to_CBase)
        self.target_in_left_cam = self.convert_from_base_2_img(self.CAM_1,self.target_3d_pos)
        self.target_in_right_cam = self.convert_from_base_2_img(self.CAM_2,self.target_3d_pos)
        
        state = self.observation()
        
        info = {}

        bolt_3d = np.matmul(self.CAM_base, self.bolt_PCs)
        reward = self.reward_func(bolt_3d)
        # assert self.observation_space.contains(state) == True

        if reward >= -0.2:
            done = True
        else:
            done = False
        
        return state, reward, done, info
    
    def observation(self):
        self.track_vec_left = self.target_in_left_cam - self.bolt2_in_left_cam
        self.track_vec_right = self.target_in_right_cam - self.bolt2_in_right_cam

        self.track_vec_left = self.track_vec_left/np.linalg.norm(self.track_vec_left,2)
        self.track_vec_right = self.track_vec_right/np.linalg.norm(self.track_vec_right,2)

        self.vec_diff_left = - self.track_vec_left + self.bolt_vec_left
        self.vec_diff_right = - self.track_vec_right + self.bolt_vec_right

        obs_left = np.matmul(np.linalg.inv(self.coord_mat_left), self.vec_diff_left.reshape(2,1))
        obs_right = np.matmul(np.linalg.inv(self.coord_mat_right), self.vec_diff_right.reshape(2,1))

        state = np.concatenate((obs_left.T, obs_right.T),axis=1).reshape(1,4)
        state = np.clip(state,-np.ones((1,4)),np.ones((1,4)))
        return state

    def reward_func(self,bolt_3d):
        direction = bolt_3d[:3,0] - bolt_3d[:3,1]
        dist = np.linalg.norm(np.cross(bolt_3d[:3,1].T-self.target_3d_pos.T,direction),2)/np.linalg.norm(direction,2)
        r = -80*dist + 1
        r = np.clip(r,-20,1)
        return r
    
    def get_rot_mat(self,xyz_rot,random='y'):
        rad = math.pi/180
        if random=='y':
            x_rot = rand.uniform(-xyz_rot[0],xyz_rot[0])/2
            y_rot = rand.uniform(-xyz_rot[1],xyz_rot[1])/2
            z_rot = rand.uniform(-xyz_rot[2],xyz_rot[2])/2
        elif random=='n':
            x_rot = xyz_rot[0]
            y_rot = xyz_rot[1]
            z_rot = xyz_rot[2]

        z_rot_mat = np.array([[math.cos(rad*z_rot),-math.sin(rad*z_rot),0],\
                        [math.sin(rad*z_rot),math.cos(rad*z_rot),0],\
                        [0,0,1]])

        y_rot_mat = np.array([[math.cos(rad*y_rot),0,math.sin(rad*y_rot)],\
                        [0,1,0],\
                        [-math.sin(rad*y_rot),0,math.cos(rad*y_rot)]])

        x_rot_mat = np.array([[1,0,0],\
                        [0,math.cos(rad*x_rot),-math.sin(rad*x_rot)],\
                        [0,math.sin(rad*x_rot),math.cos(rad*x_rot)]])

        rot_mat = np.matmul(np.matmul(x_rot_mat,y_rot_mat),z_rot_mat)
        return rot_mat
    
    def create_bolt(self):
        PC_near = np.array([[0,0,(rand.uniform(0,5)+8)/100,1]])
        near_point = np.matmul(self.CAM_base,PC_near.T)[0:-1,:]
        PC_far = np.array([[rand.uniform(-3,3)/100,rand.uniform(-3,3)/100,(rand.uniform(0,8)+15)/100,1]])
        far_point = np.matmul(self.CAM_base,PC_far.T)[0:-1,:]
        bolt_3d_pos = np.concatenate((near_point,far_point),axis=1)
        bolt_PCs = np.concatenate((PC_near.T,PC_far.T),axis=1)
        return bolt_3d_pos, bolt_PCs
    
    def create_target(self):
        target = np.matmul(self.CAM_base,np.array([[rand.uniform(-12,12),rand.uniform(-12,12),rand.uniform(0,10)+25,1]]).T/100)[0:-1,:]
        return target
    
    def get_img_pos(self,pos):
        x = 0.5 - 0.5*pos[0]/(pos[2]*math.tan(45*math.pi/180))
        y = 0.5 + 0.5*pos[1]/(pos[2]*math.tan(45*math.pi/180))
        img_pos = [x,y]
        return np.array(img_pos)

    def convert_from_base_2_img(self,CAM_pos,d3_pos):
        d3_pos = d3_pos.reshape(3,1)
        points_pos = np.concatenate((d3_pos,[[1]]),axis=0)
        pos = np.matmul(np.linalg.inv(CAM_pos),points_pos)
        img_pos = self.get_img_pos(pos)
        return img_pos.reshape(1,2)
    
    def get_included_angle(self,points1,points2):
        k1 = (points1[0,3] - points1[0,1])/(points1[0,2] - points1[0,0])
        k2 = (points2[0,3] - points2[0,1])/(points2[0,2] - points2[0,0])
        included_angle = 180/math.pi*math.atan(abs((k2 - k1)/(1 + k1*k2)))
        return included_angle
    
    def get_coord_mat(self):
        cam_base_x1 = copy.deepcopy(self.CAM_base)
        cam_base_y1 = copy.deepcopy(self.CAM_base)
        cam_base_x1[0,3] = cam_base_x1[0,3] + 0.01
        cam_base_y1[1,3] = cam_base_y1[1,3] + 0.01

        CAM_1_x1 = np.matmul(cam_base_x1,self.CAM1_to_CBase)
        CAM_1_y1 = np.matmul(cam_base_y1,self.CAM1_to_CBase)
        CAM_2_x1 = np.matmul(cam_base_x1,self.CAM2_to_CBase)
        CAM_2_y1 = np.matmul(cam_base_y1,self.CAM2_to_CBase)

        left_img_pos_x1 = self.convert_from_base_2_img(CAM_1_x1,self.target_3d_pos) - self.target_in_left_cam
        left_img_pos_y1 = self.convert_from_base_2_img(CAM_1_y1,self.target_3d_pos) - self.target_in_left_cam
        right_img_pos_x1 = self.convert_from_base_2_img(CAM_2_x1,self.target_3d_pos) - self.target_in_right_cam
        right_img_pos_y1 = self.convert_from_base_2_img(CAM_2_y1,self.target_3d_pos) - self.target_in_right_cam
        
        Cmat_L = np.concatenate((left_img_pos_x1.T,left_img_pos_y1.T),axis=1)*100
        Cmat_R = np.concatenate((right_img_pos_x1.T,right_img_pos_y1.T),axis=1)*100

        return Cmat_L, Cmat_R

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed   

    def render(self):
        return

    @property
    def observation_space(self):
        return self._observation_space
    @property
    def action_space(self):
        return self._action_space