import math
import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt


class Billboard:
    def __init__(self, score, index: int):
        self.score = score
        self.index = index


class Population:
    def __init__(self, func, dimension, SGC, TGC, IC, Xmaxs, Xmins, omega):
        self.omega = omega
        self.SGC, self.TGC, self.IC = SGC, TGC, IC
        self.Xmaxs = Xmaxs
        self.Xmins = Xmins
        self.func = func
        self.dimension = dimension

    def initialization(self):
        group = np.ndarray(shape=(self.IC, self.dimension - 1), dtype='float32')
        group[0, :] = np.random.uniform(self.Xmins, self.Xmaxs, size=(self.dimension - 1))

        for i in range(1, group.shape[0]):
            group[i, :] = group[0, :] + np.random.normal(0, self.omega, group.shape[1])

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

    def similar_taxis(self, glob: List[Billboard], sg: np.ndarray, sloc: List[List[Billboard]], tg: np.ndarray, tloc: List[List[Billboard]]):
        def temp(g: np.ndarray, loc, t):
            for i in range(g.shape[0]):
                g[i, 0, :] = g[i, loc[i][0].index, :]

                for k in range(1, g.shape[1]):
                    g[i, k, :] = g[i, 0, :] + np.random.normal(0, self.omega, g.shape[2])

                loc[i] = self.form_local_billboard(g[i])
                glob[i + t].score = loc[i][0].score

        temp(sg, sloc, 0)
        temp(tg, tloc, sg.shape[0])

    def dissimilation(self, glob: List[Billboard], sg: np.ndarray, sloc: List[List[Billboard]], tg: np.ndarray, tloc: List[List[Billboard]]):
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

    def find_max(self, gbb: List[Billboard]):
        return max(gbb, key=lambda x: x.score).score

    def find_arg(self, gbb: List[Billboard], sbb: List[List[Billboard]], sg: List[list]):
        index = max(gbb, key=lambda x: x.score).index
        return sg[index][sbb[index][0].index]

    def form_global_billboard(self, sloc: List[List[Billboard]], tloc: List[List[Billboard]]):
        glob = [Billboard(sloc[i][0].score, i) for i in range(len(sloc))]

        for i in range(len(sloc), len(sloc) + len(tloc)):
            glob.append(Billboard(tloc[i - len(sloc)][0].score, i - len(sloc)))

        return glob


def work(population):
    try:
        SG = np.ndarray(shape=(population.SGC, population.IC, population.dimension - 1), dtype='float32')
        TG = np.ndarray(shape=(population.TGC, population.IC, population.dimension - 1), dtype='float32')
        SGLBB = []
        TGLBB = []

        for i in range(SG.shape[0]):
            SG[i, :, :] = population.initialization()
            SGLBB.append(population.form_local_billboard(SG[i, :, :]))

        for i in range(TG.shape[0]):
            TG[i, :, :] = population.initialization()
            TGLBB.append(population.form_local_billboard(TG[i, :, :]))

        GLBB = population.form_global_billboard(SGLBB, TGLBB)

        old_score = population.find_max(GLBB)
        new_score = float('inf')
        count = 0

        while new_score > old_score:
            if count > 0:
                old_score = new_score

            population.similar_taxis(GLBB, SG, SGLBB, TG, TGLBB)
            population.dissimilation(GLBB, SG, SGLBB, TG, TGLBB)

            new_score = population.find_max(GLBB)
            count += 1

        if np.isnan(new_score) or np.isinf(new_score):
            print('В функции был найден разрыв. Проверьте область определения. Ошибка')
            return None
        else:
            x = population.find_arg(GLBB, SGLBB, SG)
            print(f'Максимум функции: {new_score} в точке: {x}')
            print(f'Количество итераций: {count}')


            return (x, new_score), count

    except MemoryError:
        print('Популяция слишком велика. Произошло переполнение памяти.')


f = lambda x1, x2: x1 * np.sin(x2)

population = Population(f, 3, 10, 10, 10, np.array([10, 10]), np.array([-10, -10]), 0.1)

point, count = work(population)

x1 = np.linspace(-10, 10, 200)
x2 = np.linspace(-10, 10, 200)
# plt.plot(x, f(x), 'r')
# plt.plot(point[0],point[1], 'oy')
plt.
plt.show()
