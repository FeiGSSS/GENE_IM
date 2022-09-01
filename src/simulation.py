# -*- encoding: utf-8 -*-
'''
@File    :   simulation.py
@Time    :   2022/08/25 17:05:44
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import random
from typing import List
from cana.boolean_network import BooleanNetwork
from cana.utils import binstate_to_statenum

def network_step(Net, config, pinned_vars={}):
    
    #NOTE: don't use a deepcopy or it will duplicate keys
    new_step={node:config[node] for node in config}
    
    for node in config:
        if not node.name in pinned_vars: 
            input_str = ''.join([str(config[Net.nodes[int(i)]]) for i in node.inputs])
            new_step[node]=node.outputs[binstate_to_statenum(input_str)]
    return new_step

def run_network_dynamics(N,
                         seeds,
                         pinned_vars={},
                         time_limit=10,
                         break_early=True):
        
    node2status = {s[:-2]:int(s[-1]) for s in seeds}
    # booleanNode 2 init status
    config = {node:node2status[node.name] for node in N.nodes}
    
    diffusion={0: {x.name+'-'+str(config[x]) for x in config}}
    for t in range(1, time_limit):
        config = network_step(N,
                              config,
                              pinned_vars=pinned_vars)
        new_step = {x.name+'-'+str(config[x]) for x in config}
        if new_step == diffusion[t-1] and break_early: 
            break #we have found a steady-state
        else:
            diffusion[t] = new_step #else, keep iterating
            
    return diffusion

def run_simulations(ND:BooleanNetwork,
                seeds:List,
                runs=100,
                iterations=10,
                unknown_prob=0.5):
    
    act_prob_sim = {}
    nodes = {node.name for node in ND.nodes}
    for seed in seeds: 
        seed = tuple(seed)
        
        # if len(seed) == 0:
        #     act_prob_sim[seed] = {i:{node: 0.5 for node in nodes} for i in range(iterations)}
        #     continue
            
        # make sure that one node only assigned with one inital status
        assert len(set([s[:-2] for s in seed])) == len(seed)
        act_prob_sim[seed] = {i:{node: 0.0 for node in nodes} for i in range(iterations)}

        #get the ground truth for the seed based on several runs
        for _ in range(runs):
            # initiate a random condition
            alt_seed = list(seed)
            for node in nodes - set([s[:-2] for s in alt_seed]):
                if random.random() < unknown_prob:
                    alt_seed.append(node+'-1')
                else:
                    alt_seed.append(node+'-0')

            diffusion = run_network_dynamics(ND,
                                             alt_seed,
                                             pinned_vars=[s[:-2] for s in seed],
                                             time_limit=iterations,
                                             break_early=False)
            
            for t in range(iterations):
                for node in diffusion[t]:
                    if int(node[-1]) == 1:
                        act_prob_sim[seed][t][node[:-2]] += 1

        act_prob_sim[seed] = {index:{node:act_prob_sim[seed][index][node]/runs 
                                     for node in act_prob_sim[seed][index]} 
                              for index in range(iterations)}

    return act_prob_sim