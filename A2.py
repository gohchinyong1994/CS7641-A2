# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:59:34 2024

@author: User
"""
import numpy as np 
import pandas as pd
import mlrose_hiive
import time
import matplotlib.pyplot as plt
import dataframe_image
import os

os.chdir(r"D:\Georgia Tech\CS7641 Machine Learning\A2")

SEED = 1337
MAX_ATTEMPTS = 500
MAX_ITERS = 1000

np.random.seed(1337)

def extendCurve(curve, iters):
    ''' extend the fitness curve for a run given it ends before max iterations'''
    if len(curve) < iters:
        return [x[0] for x in curve] + [curve[-1][0] for _ in range(iters - len(curve))]
    else:
        return [x[0] for x in curve]


''' Fitness Function is sourced from
https://mlrose.readthedocs.io/en/stable/source/tutorial1.html#what-is-an-optimization-problem
'''
# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

   # Initialize counter
   fitness_cnt = 0

   # For all pairs of queens
   for i in range(len(state) - 1):
       for j in range(i + 1, len(state)):

        # Check for horizontal, diagonal-up and diagonal-down attacks
        if (state[j] != state[i]) \
            and (state[j] != state[i] + (j - i)) \
            and (state[j] != state[i] - (j - i)):

            # If no attacks, then increment counter
           fitness_cnt += 1
   return fitness_cnt


def runProblem(label, problem, rhc_kwargs, sa_kwargs, ga_kwargs, mm_kwargs, MAX_ITERS, to_run='all'):   
    
    stats = []
    
    if to_run == 'all' or 'rhc' in to_run:
        # Random Hill Climb
        start_time = time.time()
        rhc_bs, rhc_f, rhc_curve  = mlrose_hiive.random_hill_climb(
            problem, 
            max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERS, random_state = SEED, curve=True,
            **rhc_kwargs,
            )
        
        end_time = time.time()
        rhc_time = end_time-start_time
        rhc_it = len(rhc_curve)
        stats.append(['RHC', rhc_f, rhc_it, rhc_time])
        print('RHC: Best Fitness: %.3f Time Taken: %.3f' % (rhc_f, rhc_time))
        rhc_curve = extendCurve(rhc_curve, MAX_ITERS)
    
    if to_run == 'all' or 'sa' in to_run:
    # Simulated Annealing
        start_time = time.time()
        sa_bs, sa_f, sa_curve  = mlrose_hiive.simulated_annealing(
            problem, 
            max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERS ,random_state = SEED, curve=True,
            **sa_kwargs,
            )
        end_time = time.time()
        sa_time = end_time-start_time
        sa_it = len(sa_curve)
        stats.append(['SA', sa_f, sa_it, sa_time])
        print('SA: Best Fitness: %.3f Time Taken: %.3f' % (sa_f, sa_time))
        sa_curve = extendCurve(sa_curve, MAX_ITERS)
    
    # Genetic Algorithm
    if to_run == 'all' or 'ga' in to_run:
        start_time = time.time()
        ga_bs, ga_f, ga_curve  = mlrose_hiive.genetic_alg(
            problem,
            max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERS ,random_state = SEED, curve=True,
            **ga_kwargs,
            )
        
        end_time = time.time()
        ga_time = end_time-start_time
        ga_it = len(ga_curve)
        stats.append(['GA', ga_f, ga_it, ga_time])
        print('GA: Best Fitness: %.3f Time Taken: %.3f' % (ga_f, ga_time))
        ga_curve = extendCurve(ga_curve, MAX_ITERS)
    
    if to_run == 'all' or 'mm' in to_run:
        # MIMIC
        start_time = time.time()
        mm_bs, mm_f, mm_curve  = mlrose_hiive.mimic(
            problem,
            max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERS ,random_state = SEED, curve=True,    
            **mm_kwargs,
            )
        
        end_time = time.time()
        mm_time = end_time-start_time
        mm_it = len(mm_curve)
        stats.append(['MM', mm_f, mm_it, mm_time])
        print('MIMIC: Best Fitness: %.3f Time Taken: %.3f' % (mm_f, mm_time))
        mm_curve = extendCurve(mm_curve, MAX_ITERS)
    
    iters = list(range(1, MAX_ITERS+1))
    
    stats_df = pd.DataFrame( stats, columns = ('Algorithm', 'Best Fitness', 'Iterations', 'Time Taken (s)'))
    dataframe_image.export(stats_df,"charts/stats_%s.png" % label)
    
    plt.figure()
    
    if to_run == 'all' or 'rhc' in to_run:
        plt.plot(iters, rhc_curve, label='RHC', color='green')
    if to_run == 'all' or 'sa' in to_run:
        plt.plot(iters, sa_curve, label='SA', color='blue')
    if to_run == 'all' or 'ga' in to_run:
        plt.plot(iters, ga_curve, label='GA', color='red')
    if to_run == 'all' or 'mm' in to_run:
        plt.plot(iters, mm_curve, label='MIMIC', color='purple')
    plt.legend()
    plt.title('Fitness Curve: %s' % label)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.savefig("charts/fitness_%s.png" % label)
    
    
rhc_kwargs = { 'restarts'       : 5}
sa_kwargs  = { 'schedule'       : mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.5, min_temp = 0.001)}
ga_kwargs  = { 'pop_size'       : 200,
               'mutation_prob'  : 0.2,
               }
mm_kwargs  = { 'pop_size'       : 200,
               'keep_pct'       : 0.2
               }
# N-Queens 
fitness = mlrose_hiive.CustomFitness(queens_max)
problem = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=True, max_val=8)
problem.set_mimic_fast_mode(True)
runProblem('N-Queens', problem, rhc_kwargs, sa_kwargs, ga_kwargs, mm_kwargs, MAX_ITERS)

# FourPeaks
fitness = mlrose_hiive.FourPeaks()
problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
problem.set_mimic_fast_mode(True)
runProblem('FourPeaks', problem, rhc_kwargs, sa_kwargs, ga_kwargs, mm_kwargs, MAX_ITERS)

# OneMax
fitness = mlrose_hiive.OneMax()
problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)
problem.set_mimic_fast_mode(True)
runProblem('OneMax', problem, rhc_kwargs, sa_kwargs, ga_kwargs, mm_kwargs, MAX_ITERS)


# FourPeaks - 10000 Iters RHC v SA
fitness = mlrose_hiive.FourPeaks()
problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
problem.set_mimic_fast_mode(True)
runProblem('FourPeaks-10000 Iters', problem, rhc_kwargs, sa_kwargs, ga_kwargs, mm_kwargs, MAX_ITERS=10000, to_run=('rhc','sa'))


# FourPeaks - 10000 Iters RHC v SA
fitness = mlrose_hiive.FourPeaks()
problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
problem.set_mimic_fast_mode(True)
sa_kwargs2  = { 'schedule'       : mlrose_hiive.GeomDecay(100)}
runProblem('FourPeaks-10000 Iters-GeomDecay(100)', problem, rhc_kwargs, sa_kwargs2, ga_kwargs, mm_kwargs, MAX_ITERS=10000, to_run=('rhc','sa'))

# OneMax - 2000 Iters RHC v SA
fitness = mlrose_hiive.OneMax()
problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)
problem.set_mimic_fast_mode(True)
sa_kwargs2  = { 'schedule'       : mlrose_hiive.GeomDecay(100)}
runProblem('OneMax-2000 Iters-GeomDecay(100)', problem, rhc_kwargs, sa_kwargs2, ga_kwargs, mm_kwargs, MAX_ITERS=2000, to_run=('rhc','sa'))


