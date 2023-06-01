import math
import random

import numpy as np


class Omega:
    omega_changed = False
    new_omega = 0.0

    def __init__(self):
        self.omega = 0.0
        self.next_number = 0.0
        self.if_next_number = False

    def new_number_with_omega(self):
        if self.if_next_number:
            self.if_next_number = False
            return self.next_number * self.omega
        else:
            s, x, y = 0, None, None

            while s >= 1 or s == 0:
                x = 2 * random.random() - 1
                y = 2 * random.random() - 1
                s = x * x + y * y

            part = math.sqrt(-2 * math.log(s, math.e) / s)
            self.next_number = y * part
            self.if_next_number = True

            return x * part * self.omega


class Billboard:
    def __init__(self, score, index: int):
        self.score = score
        self.index = index


class Population:
    def __init__(self, func, dimension):
        self.omega = Omega()
        self._SGC, self._TGC, self._IC = np.uint32(), np.uint32(), np.uint32()
        self._xmaxs = None
        self.Xmins = None
        self.func = func
        self.dimension = dimension

    @property
    def SGC(self):
        return self._SGC

    @SGC.setter
    def SGC(self, value):
        if value == 0:
            raise Exception('Error')
        self._SGC = value

    @property
    def TGC(self):
        return self._TGC

    @TGC.setter
    def TGC(self, value):
        if value == 0:
            raise Exception('Error')
        self._TGC = value

    @property
    def IC(self):
        return self._IC

    @IC.setter
    def IC(self, value):
        if value == 0:
            raise Exception('Error')
        self._IC = value

    @property
    def Xmaxs(self):
        return self._xmaxs

    @Xmaxs.setter
    def Xmaxs(self, value):
        for v in value:
            if v < self._xmaxs:
                raise Exception('Error')
        self._xmaxs = value

    def sides(self, x, dimension_number: int):
        if x > self.Xmaxs[dimension_number]:
            x = self.Xmaxs[dimension_number]
        elif x < self.Xmins[dimension_number]:
            x = self.Xmins[dimension_number]
        return x

    def initialization(self):
        group = np.ndarray(shape=(self.IC, self.dimension - 1), dtype=np.float)
        group[0] = self.Xmins + (self.Xmaxs - self.Xmins) * np.random.uniform(size=(self.dimension - 1))

        for i in range(1, group.shape[0]):
            for j in range(group.shape[1]):
                group[i, j] = self.sides(group[0, j] + self.omega.new_number_with_omega(), j)

        return group

    def form_local_billboard(self, group: np.ndarray):
        local = []
        max = self.func(group[0, :])
        index = 0
        flag = False

        for i in range(group.shape[0]):
            local.append(Billboard(self.func(group[i, :]), i))

            if local[i].score > max:
                if i != 0:
                    flag, index, max = True, i, local[i].score

        if flag:
            local[index].score = local[0].score
            local[0].score = max
            local[index].index = local[0].index
            local[0].index = index

        return local

    def similar_taxis(self, glob: list[Billboard], sg: np.ndarray, sloc: list[list[Billboard]], tg: np.ndarray, tloc: list[list[Billboard]]):
        def temp(g: np.ndarray, loc, t):
            for i in range(g.shape[0]):
                g[i, 0, :] = g[i, loc[i][0].index, :]

                for k in range(1, g.shape[1]):
                    for j in range(g.shape[2]):
                        g[i, k, j] = self.sides(g[i, 0, j] + self.omega.new_number_with_omega(), j)

                loc[i] = self.form_local_billboard(g[i])
                glob[i + t].score = loc[i][0].score

        temp(sg, sloc, 0)
        temp(tg, tloc, sg.shape[0])

    def dissimilation(self, glob: list[Billboard], sg: np.ndarray, sloc: list[list[Billboard]], tg: np.ndarray, tloc: list[list[Billboard]]):
        for i in range(sg.shape[0]):
            for j in range(sg.shape[0], len(glob)):
                if glob[j].score > glob[i].score:
                    sg[glob[i].index, :, :], tg[glob[j].index, :, :] = tg[glob[j].index, :, :], sg[glob[i].index, :, :]
                    sloc[glob[i].index], tloc[glob[j].index] = tloc[glob[j].index], sloc[glob[i].index]
                    glob[i].score, glob[j].score = glob[j].score, glob[i].score

                    if min(glob, key=lambda x: x.score) == glob[i].score:
                        sg[glob[i].index, :, :] = self.initialization()
                        sloc[glob[i].index] = self.form_local_billboard(sg[glob[i].index])
                        glob[i].score = sloc[glob[i].index][0].score

    def find_max(self, gbb: list[Billboard]):
        return max(gbb, key=lambda x: x.score).score

    def find_arg(self, gbb: list[Billboard], sbb: list[list[Billboard]], sg: list[list]):
        index = max(gbb, key=lambda x: x.score).index
        return sg[index][sbb[index][0].index]

    def form_global_billboard(self, sloc: list[list[Billboard]], tloc: list[list[Billboard]]):
        glob = [Billboard(sloc[i][0].score, i) for i in range(len(sloc))]

        for i in range(len(sloc), len(sloc) + len(tloc)):
            glob.append(Billboard(tloc[i - len(sloc)][0].score, i - len(sloc)))

        return glob
