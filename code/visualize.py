# visualize.py
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

Colors = ['green', 'blue', 'orange']


class Animation:
    def __init__(self, my_map, starts, goals, paths):
        self.my_map = np.flip(np.transpose(my_map), 1)
        def conv(p): return (p[1], len(self.my_map[0]) - 1 - p[0])
        self.starts = [conv(s) for s in starts]
        self.goals  = [conv(g) for g in goals]
        self.paths  = [[conv(l) for l in p] for p in paths] if paths else []

        aspect = len(self.my_map) / len(self.my_map[0])
        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.patches, self.artists, self.agents, self.names = [], [], {}, {}
        x_min, y_min = -0.5, -0.5
        x_max, y_max = len(self.my_map) - 0.5, len(self.my_map[0]) - 0.5
        plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)

        self.patches.append(Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                      facecolor='none', edgecolor='gray'))

        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                if self.my_map[i][j]:
                    self.patches.append(Rectangle((i-0.5, j-0.5), 1, 1,
                                                  facecolor='gray', edgecolor='gray'))

        self.T = max((len(p) for p in self.paths), default=0) - 1

        for i, g in enumerate(self.goals):
            self.patches.append(Rectangle((g[0]-0.25, g[1]-0.25), 0.5, 0.5,
                                          facecolor=Colors[i%len(Colors)], edgecolor='black', alpha=0.5))

        for i in range(len(self.paths)):
            col = Colors[i%len(Colors)]
            self.agents[i] = Circle(self.starts[i], 0.3, facecolor=col, edgecolor='black')
            self.agents[i].original_face_color = col
            self.patches.append(self.agents[i])
            txt = self.ax.text(self.starts[i][0], self.starts[i][1]+0.25, str(i),
                               ha='center', va='center')
            self.names[i] = txt
            self.artists.append(txt)

        self.anim = animation.FuncAnimation(self.fig, self._animate,
                                            init_func=self._init,
                                            frames=int(self.T+1)*10,
                                            interval=100, blit=True)

    def save(self, fname, speed=1):
        self.anim.save(fname, fps=10*speed, dpi=200,
                       savefig_kwargs={"pad_inches":0, "bbox_inches":"tight"})

    @staticmethod
    def show(): plt.show()

    def _init(self):
        for p in self.patches: self.ax.add_patch(p)
        for a in self.artists: self.ax.add_artist(a)
        return self.patches + self.artists

    def _animate(self, frame):
        t = frame / 10.0
        for i, path in enumerate(self.paths):
            pos = self._interp(t, path)
            self.agents[i].center = pos
            self.names[i].set_position((pos[0], pos[1]+0.5))

        for a in self.agents.values():
            a.set_facecolor(a.original_face_color)

        pos = [np.array(a.center) for a in self.agents.values()]
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                if np.linalg.norm(pos[i] - pos[j]) < 0.7:
                    self.agents[i].set_facecolor('red')
                    self.agents[j].set_facecolor('red')
        return self.patches + self.artists

    @staticmethod
    def _interp(t, path):
        if t <= 0: return np.array(path[0])
        if t >= len(path)-1: return np.array(path[-1])
        i = int(t)
        frac = t - i
        return (1-frac)*np.array(path[i]) + frac*np.array(path[i+1])