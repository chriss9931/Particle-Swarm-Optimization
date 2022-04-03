'''
Single Objective Particle Swarm Optimization Algorithm
    Takes input:
        efunc - Evaluation function that matches the # of DVs as the initial population
        cfunc - contrstaint function that checks
        ipop - Initial population: n x m array with n number of starting points and m number of design variables
        LB - 1 x m array that provides lower bounds
        UB - 1 x m array that provides upper bounds
        Nmx - Maximum number of iterations/generations
        alpha - velocity weight
        A - Personal Best weight
        B - Global Best weight
'''
import numpy as np


def pso(efunc, cfunc, ipop, LB, UB, Nmx, alpha, A, B):
    pop = np.empty((len(ipop)), dtype=object)
    for i in range(len(ipop)):
        pop[i] = Particle(efunc, cfunc, ipop[i], LB, UB)
    fg_best = 1E15
    xg_best = None
    for particle in pop:
        f_local, x_local = particle.get_best()
        if f_local < fg_best:
            fg_best = f_local
            xg_best = x_local
            
    positions = np.zeros((Nmx, len(ipop), len(LB)))
    positions[0, :, :] = ipop
    fg_best_changeITER = 0
    print('initialized')
    N = 1
    while N < Nmx:
        if N % 100 == 0:
            print('Iteration #', N)
        for particle in pop:
            particle.Move(xg_best, alpha, A, B)
        fg_best_prev = fg_best
        for q in range(len(pop)):
            positions[N, q, :] = pop[q].get_current()[1]
        for particle in pop:
            f_local, x_local = particle.get_best()
            if (f_local < fg_best) and particle.elig():
                fg_best = f_local
                xg_best = x_local
        if fg_best == fg_best_prev:
            fg_best_changeITER += 1
        else:
            fg_best_changeITER = 0
        if fg_best_changeITER > 20:
            print('STOP: Best function evaluation did not change for 10 iterations')
            break
        N += 1
    fg_best = efunc(xg_best)
    f_pop = np.zeros((len(pop), len(LB)))
    f_pop_best = np.zeros((len(pop), len(LB)))
    x_pop = np.zeros((len(pop), len(LB)))
    x_pop_best = np.zeros((len(pop), len(LB)))
    for p in range(len(pop)):
        f_pop[p], x_pop[p] = pop[p].get_current()
        f_pop_best[p], x_pop_best[p] = pop[p].get_best()

    return N, fg_best, xg_best, f_pop, f_pop_best, x_pop, x_pop_best, positions


''' Define particle class to simplify keeping track of particles'''
class Particle():
    def __init__(self, func, constraintfunc, position, LB, UB):
        self.func = func
        self.constraintfunc = constraintfunc
        self.x_current = position
        self.v_current = np.zeros_like(position)
        self.f_current, self.eligible = self.evaluate(position)
        self.fp_best = self.f_current
        self.xp_best = np.zeros_like(position)
        self.LB = LB
        self.UB = UB
        
    def evaluate(self, x):
        f_vals = self.func(x)
        eligible = self.constraintfunc(f_vals, x)
        return f_vals, eligible
    
    def elig(self):
        return self.eligible
    
    def get_best(self):
        return self.fp_best, self.xp_best
    
    def get_current(self):
        return self.f_current, self.x_current
    
    def Move(self, xg_best, alpha, A, B):
        Vi = (alpha*self.v_current
             + A*np.random.rand()*(self.xp_best - self.x_current)
             + B*np.random.rand()*(xg_best - self.x_current))
        X = self.x_current + Vi
        self.x_current = X
        # Check for new positions outside of the bounds
        for variable in range(len(self.x_current)):
            if self.x_current[variable] < self.LB[variable]:
                self.x_current[variable] = self.LB[variable]
            if self.x_current[variable] > self.UB[variable]:
                self.x_current[variable] = self.UB[variable]
        # re-evaluate functions
        self.f_current, self.eligible = self.evaluate(self.x_current)
        if self.eligible:
            if self.f_current < self.fp_best:
                self.fp_best = self.f_current
                self.xp_best = self.x_current
