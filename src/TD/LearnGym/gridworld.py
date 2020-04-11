""" Create Your Own GridWorld Environment
"""
import gym
from gym import spaces
from gym.envs.classic_control import rendering


class Grid(object):
    """ Individual Basic Gird
    """

    def __init__(self, x, y, default_mode=0, u_size=40):
        self.x = x            # coordinate-x
        self.y = y            # coordinate-y
        self._mode = default_mode  # grid mode, 0 or 1: normal, 2: end, 3: obs
        self.u_size = u_size  # size, defaultly height equals width

    @property
    def reward(self):
        if self._mode == 0 or self._mode == 1:  # normal or start
            self.__reward = -1
        elif self._mode == 2:  # end
            self.__reward = 0
        elif self._mode == 3:  # obstacle
            self.__reward == -100
        else:
            pass
        return self.__reward


class GridWorldEnv(gym.Env):
    """Gri World Environment
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 u_size=40,
                 actions=4,
                 is_list=None,
                 n_width=10,
                 n_height=7,
                 start=(3, 1),
                 ends=[(3, 9)],
                 obs=[(4, 3),(4,4),(4,5),(4,6)]):
        # super().__init__()
        self.u_size = u_size        # size of a gird
        self.is_list = is_list      # if grid is given by a 2D list
        if is_list:
            self.n_width = len(is_list)      # num of grids horizontally
            self.n_height = len(is_list[0])    # num of grids vertically
        else:
            self.n_width = n_width
            self.n_height = n_height
            self.start = start
            self.ends = ends
            self.obs = obs
        self.action_space = spaces.Discrete(actions)  # total number of actions
        self.observation_space = spaces.Discrete(self.n_width*self.n_height)
        # self.reset()
        self.viewer = None

    def reset(self):
        self.grids = []
        if self.is_list:
            for x in range(self.n_height):
                one_row = []
                for y in range(self.n_width):
                    one_row.append(
                        Grid(x, y, self.is_list[x][y], u_size=self.u_size))
                self.grids.append(one_row)
        else:
            for x in range(self.n_height):
                one_row = []
                for y in range(self.n_width):
                    one_row.append(Grid(x, y, u_size=self.u_size))
                self.grids.append(one_row)
            self.grids[self.start[0]][self.start[1]]._mode = 1
            for x, y in self.ends:
                self.grids[x][y]._mode = 2
            for x, y in self.obs:
                self.grids[x][y]._mode = 3
        self.state = self._xy_to_state(self.start[0], self.start[1])

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.action = action    # action for rendering
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y
        # action effect
        new_x, new_y = self._action_effect(new_x, new_y, action)
        # boundary effect
        new_x, new_y = self._boundary_effect(new_x, new_y)

        # obs effect:
        if (new_x, new_y) in self.obs:
            new_x, new_y = old_x, old_y

        reward = self.grids[new_x][new_y].reward
        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return self.state, reward, done, info

    def render(self,
               mode='human',
               close=False):
        if close:
            if not self.viewer:
                self.viewer.close()
                self.viewer = None
            return
        else:
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.n_width*self.u_size,
                                               self.n_height*self.u_size)
            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
            for i in range(self.n_width+1):
                line = rendering.Line(start = (i*u_size, 0),
                                      end =(i*u_size, u_size*self.n_height))
                line.set_color(0.5,0,0)
                self.viewer.add_geom(line)
            for i in range(self.n_height):
                line = rendering.Line(start = (0, i*u_size),
                                      end = (u_size*self.n_width, i*u_size))
                line.set_color(0,0,1)
                self.viewer.add_geom(line)
            '''
            # draw grids
            for x in range(self.n_height):
                for y in range(self.n_width):
                    # draw rectangle
                    self._draw_rect(x, y)
                    # draw outline
                    self._draw_outline(x, y)
            # draw agent
            self.agent = rendering.make_circle(self.u_size/4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)
        # update agent loaction
        self._update_agent()
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _draw_rect(self, x, y, m=2):
        u_size = self.u_size
        x, y = y, x  # for rendering
        v = [(x*u_size+m, y*u_size+m),
             ((x+1)*u_size-m, y*u_size+m),
             ((x+1)*u_size-m, (y+1)*u_size-m),
             (x*u_size+m, (y+1)*u_size-m)]
        rect = rendering.FilledPolygon(v)
        x, y = y, x  # for judging
        if (x, y) in self.ends:
            rect.set_color(1.0, 0.0, 0.0)
        elif (x, y) in self.obs:
            rect.set_color(0.1, 0.1, 0.1)
        elif (x, y) == self.start:
            rect.set_color(0.0, 1.0, 0.0)
        else:
            rect.set_color(0.8, 0.8, 0.8)
        self.viewer.add_geom(rect)

    def _draw_outline(self, x, y, m=2):
        u_size = self.u_size
        x, y = y, x
        v_outline = [(x*u_size+m, y*u_size+m),
                     ((x+1)*u_size-m, y*u_size+m),
                     ((x+1)*u_size-m, (y+1)*u_size-m),
                     (x*u_size+m, (y+1)*u_size-m)]
        outline = rendering.make_polygon(v_outline, False)
        outline.set_linewidth(3)
        x, y = y, x
        if self._is_end_state(x, y):
            # ends
            outline.set_color(1.0, 0.0, 0.0)
        elif (x, y) == self.start:
            # start
            outline.set_color(0.0, 1.0, 0.0)
        else:
            pass
        self.viewer.add_geom(outline)

    def _is_end_state(self, x, y):
        return True if (x, y) in self.ends else False

    def _state_to_xy(self, s):
        y = s % self.n_width
        x = int((s - y) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y):
        return y + self.n_width * x

    def _action_effect(self, x, y, action):
        new_x, new_y = x, y
        if action == 0:
            new_y -= 1   # left
        elif action == 1:
            new_y += 1   # right
        elif action == 2:
            new_x += 1   # up
        elif action == 3:
            new_x -= 1   # down
        else:
            pass
        return new_x, new_y

    def _boundary_effect(self, x, y):
        new_x, new_y = x, y
        if new_x < 0:
            new_x = 0
        if new_x >= self.n_height:
            new_x = self.n_height-1
        if new_y < 0:
            new_y = 0
        if new_y >= self.n_width:
            new_y = self.n_width-1
        return new_x, new_y

    def _update_agent(self):
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation(
            (y+0.5)*self.u_size, (x+0.5)*self.u_size)  # translate


if __name__ == "__main__":
    env = GridWorldEnv()
    env.reset()
    env.render()
    for _ in range(1000):
        env.render()
        # left, right, up, down
        a = env.action_space.sample()
        state, reward, isdone, info = env.step(a)
        print("{0}, {1}, {2}".format(state, reward, isdone))
        # print(_)
    print("env closed")
