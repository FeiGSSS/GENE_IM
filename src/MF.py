# -*- encoding: utf-8 -*-
'''
@File    :   MF.py
@Time    :   2022/08/25 10:40:01
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

from typing import Dict, List, Set
from itertools import combinations

from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode

def find_modules(N:BooleanNetwork,
                 seed_size:int,
                 seed_unit:List,
                 seeds:list=None,
                 iterations:int=10,
                 pinning:Set=None,
                 reduced:bool=True,
                 LUT:Dict=None,
                 p:float=0.5,
                 data:bool=True,
                 pin_start:bool=True,
                 models:int=1):
    
    #define seeds
    if seeds is None:
        seeds = list(combinations(seed_unit, seed_size))

    # 存储每一个 seed 产生的结果
    modules = {seed:set() for seed in seeds}

    for seed in seeds: #ALTERNATE
        act_prob = average_seed_mf(N, seed, p, iterations, pinning, reduced, LUT, pin_start, models=models)
        
        if data:
            modules[seed] = act_prob
        else:
            raise NotImplementedError
        
    return modules

def average_seed_mf(N:BooleanNetwork,
                    seed:List,
                    p:float,
                    iterations:int,
                    pinning:Set,
                    reduced:bool,
                    LUT:Dict,
                    pin_start:bool,
                    models:int)->Dict:
    """run the IBMFA for a seed, averaging over possible update schedules
       if models=1, this replicates run_seed_mf
    """
    
    avg_prob = {i: 
        {node.name: 0.0 for node in N.nodes} 
                for i in range(iterations+1)} #average act_prob
    
    for _ in range(models):
        seed_nodes, active_prob = mf_seed(N, seed, p, iterations)
        active_prob = synchronous_mf(N, seed_nodes, active_prob, iterations, pinning, reduced, LUT, pin_start)
        avg_prob = {i:
            {node.name: avg_prob[i][node.name]+active_prob[i][node.name] for node in N.nodes}
                    for i in range(iterations+1)}
    return {i:{node.name: avg_prob[i][node.name]/models for node in N.nodes}
            for i in range(iterations+1)}

    

def mf_seed(N:BooleanNetwork, seed:List, p:float=0.5, iterations:int=10):
    # 存储MF每一个iteration每一个节点的激活概率，初始为p=0.5
    active_prob = {i: 
        {node.name: p for node in N.nodes} 
                   for i in range(iterations+1)}
    
    seed_nodes = [n[:-2] for n in seed]
    seed_state = [int(n[-1])  for n in seed]
    
    for n, s in zip(seed_nodes, seed_state):
        active_prob[0][n] = 0 if s == 0 else 1
        
    return seed_nodes, active_prob

def synchronous_mf(N:BooleanNetwork,
                   seed_nodes:List,
                   act_prob:Dict,
                   iterations:int,
                   pinning:Set,
                   reduced:bool,
                   LUT:Dict,
                   pin_start:bool):
    #synchronous update
    #pinning allows to pin a certain state (e.g. 0,1), pin_start allows to pin the seed
    #iterate through the mean-field approximation
    for i in range(1, iterations+1):
        for node in N.nodes:
            if node.name in seed_nodes and pin_start: #pin the starting nodes
                act_prob[i][node.name] = act_prob[i-1][node.name]
                continue
            
            s = mf_approx(N, node, act_prob, i, reduced=reduced, LUT=LUT) #solve equation based on probabilities of inputs
            s = max(min(s, 1), 0) #fix rounding errors
            
            #allow pinning when state is reached
            if (0 in pinning) and act_prob[i-1][node.name] == 0:
                act_prob[i][node.name] = 0.0
            elif (1 in pinning) and act_prob[i-1][node.name] == 1:
                act_prob[i][node.name] = 1.0
            else:
                act_prob[i][node.name] = s
                
    return act_prob


def mf_approx(N:BooleanNetwork,
              node:BooleanNode,
              act_prob:Dict,
              i:int,
              reduced:bool,
              LUT:Dict):
    s = 0.0
    if not node.inputs: 
        #constant with no inputs, so no way to update its state
        return act_prob[i-1][node.name]
    
    if reduced and len(LUT[node.name]) == 0:
        #this node has constant output
        if int(node.outputs[0]) == 1: 
            return 1.0
        else: 
            return 0.0
        
    for inputs, out in LUT[node.name].items(): 
        # look at each input configuration
        # solve equation based on probabilities of inputs
        # NOTE: with reduced, we look at all rows because 
        # the LUT has already been reduced
        if int(out) == 1 or reduced: 
            # 如果是reduced，则聚合所有，最后判断是否 1-s
            # 如果不是reduced，则只聚合 out = 1
            #ignore those that don't lead to the required state activation
            p = 1.0
            for j, inp in enumerate(inputs):
                j_name = N.nodes[node.inputs[j]].name
                if int(inp)==0:
                    #contribution from OFF node
                    p *= 1 - act_prob[i-1][j_name]
                else: #contribution from ON node
                    p *= act_prob[i-1][j_name]
            s += p
            
    if reduced and int(out) == 0: #only have to check one row because all row outputs are the same
        return 1-s
    else:
        return s