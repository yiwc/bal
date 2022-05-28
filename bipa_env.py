import pathlib
import gym
import typing
from typing import Tuple
from gym.spaces import Box,Dict
from gym import spaces
import mujoco_py
import os
import copy
import os
import copy
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import numpy as np
from PIL import Image


import time
import rotations
DEFAULT_SIZE=500

camera_output=pathlib.Path("camera_output")
camera_output.mkdir(exist_ok=True)

class bipa_env(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        # 可变参数

        self.cam_w=255
        self.cam_h=255
        self.max_steps=100 # 100步后，游戏重启

        # 固定参数
        self.action_space=Box(low=-1,high=1,shape=[6,])
        self.observation_space=Box(low=-1,high=1,shape=[16,])

        fullpath="xml_bipa.xml"
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=3)
        self.viewer=None

        self.viewer = mujoco_py.MjViewer(self.sim)
        self.render_offscreen = mujoco_py.MjRenderContextOffscreen(self.sim)
        self.target_objects=[0,1]
        # self.gym_viewer1=rendering.SimpleImageViewer()

        # 其他 用不上的参数
        self._cam_save_id=0
        self._steps=0

    def reset(self)->"Obs":
        
        # 物体1
        self.sim.model.site_pos[0] = np.array([0,0,0]) # 坐标，
        self.sim.model.site_quat[0] = np.array([1,0,0,0]) # 初始旋转
        self.sim.model.site_size[0] = np.array([0.01,0.01,0.1]) # 长为10cm，宽高为1cm
        # TODO， 泽栋，这里可以加初始的随机旋转、位移

        # 物体2
        self.sim.model.site_pos[1] = np.array([0,0,0.5]) # 坐标，
        self.sim.model.site_quat[1] = np.array([1,0,0,0]) # 初始旋转
        self.sim.model.site_size[1] = np.array([0.01,0.01,0.1]) # 长为10cm，宽高为1cm
        
        # 
        # self.sim.
        self.sim.model.cam_pos[0] = np.array([1,1,0])
        self.sim.model.cam_pos[1] = np.array([2,2,0])
        self.sim.model.cam_fovy[0]=150
        self.sim.model.cam_fovy[1]=150
        
        print(f"\n## self.sim.model 里 可用的参数:\n\n{dir(self.sim.model)}\n\n\n")
        print(f"\n## self.sim.data 里 可用的参数:\n\n{dir(self.sim.data)}\n\n\n")

        # 相机设置lookat还不太会，你可以看看
        # self.viewer.cam.lookat[0] += 0.5         # cx,y,z offset from the object (works if trackbodyid=-1)
        # self.viewer.cam.lookat[1] += 0.5
        # self.viewer.cam.lookat[2] += 0.5

        

        # 其他
        self._steps=0

    def step(self, action: "ActType") -> Tuple["ObsType", float, bool, dict]:
        obs=self.observation()
        assert self.action_space.contains(action)

        # 将输入的动作，转换为一个object的旋转和平移
        site_id="target0" # 我们只动target0，不动target1
        site_id = self.sim.model.site_name2id(site_id) # == 0
        pos=self.sim.model.site_pos[site_id] 
        quat=self.sim.model.site_quat[site_id] 
        pos+=action[:3]*0.01 # 平移
        quat=rotations.quat_mul(quat,rotations.euler2quat(action[-3:]*0.05)) # 旋转
        self.sim.model.site_pos[site_id]=pos #
        self.sim.model.site_quat[site_id] =quat
        self.sim.forward()

        rew=self.get_reward()
        done=self.get_done()
        self._steps+=1
        return obs,rew,done,{}

    def get_done(self):
        """
        
        """
        done= 1 if self._steps>=self.max_steps else 0
        return done

    def get_reward(self):
        """
        TODO：Zedong, 这里获取reward，可以通过获取两个object的旋转轴，然后求差。
        """
        return 0
    def observation(self):
        return self.observation_space.sample()


    def render_human(self):
        self.viewer.render()

    def render_two_camera(self)->None:

        self.render_offscreen.render(self.cam_w,self.cam_h,0)

        cam1_data = self.render_offscreen.read_pixels(self.cam_w, self.cam_h, depth=False)
        Image.fromarray(cam1_data).save(camera_output.joinpath(f"cam{self._cam_save_id}_1.jpeg"))
        
        self.render_offscreen.render(self.cam_w,self.cam_h,1)
        cam2_data = self.render_offscreen.read_pixels(self.cam_w,self.cam_h, depth=False)
        Image.fromarray(cam2_data).save(camera_output.joinpath(f"cam{self._cam_save_id}_2.jpeg"))
        self._cam_save_id+=1
        print(f"camera image has saved in {camera_output}")
        return None

if __name__=="__main__":
    pass
    env=bipa_env()
    env.reset()

    # 3D 视角、交互
    while True:
        env.render_human()
        obs,rew,done,info=env.step(env.action_space.sample())
        if done:
            env.reset()
            
    # # # 相机视角，会保存图片
    # for i in range(10):
    #     env.render_two_camera()
