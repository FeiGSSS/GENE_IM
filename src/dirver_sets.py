# -*- encoding: utf-8 -*-
'''
@File    :   dirver_sets.py
@Time    :   2022/09/01 13:43:01
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

from typing import List, Dict
from cana.boolean_network import BooleanNetwork
from src.MF import find_modules
from src.utils import config_entropy

def select_top_seed(seed_prob:Dict):
    seeds = []
    entropys = []
    for k, p in seed_prob.items():
        entropy, _ = config_entropy(p, base=2, normalized=True)
        seeds.append(k)
        entropys.append(list(entropy.values())[-1])
    return seeds[entropys.index(min(entropys))], min(entropys)

def update_candidates(candidates:List, selections:List):
    newest_seeds = selections[-1][0]
    selected_node_names = [s[:-2] for s in newest_seeds]
    assert len(set(selected_node_names)) == len(selected_node_names)
    
    new_candidates = []
    for c in candidates:
        if c[:-2] not in selected_node_names:
            new_candidates.append(c)
    
    return new_candidates
            
    

def top_selection(N:BooleanNetwork,
                  seed_unit:List,
                  reduced:bool=True,
                  LUT:Dict=None,
                  max_seed_size:int=10,
                  pinning={},
                  iterations=10,
                  unknown_prob=0.5,
                  pin_start=True,
                  models=1):

    modules = find_modules(N = N, 
                           seed_size = 1, 
                           seed_unit = seed_unit,
                           iterations = iterations,
                           pinning = pinning,
                           reduced = reduced,
                           LUT = LUT,
                           p = unknown_prob,
                           data = True,
                           pin_start = pin_start,
                           models = models)
    candidates = [str(x[0]) for x in list(modules.keys())]
    
    selections = []
    # first select
    top_seed, top_seed_entropy = select_top_seed(modules)
    # the entropy without any seed
    top_seed_entropy_full, _ = config_entropy(modules[top_seed])
    selections.append([[], top_seed_entropy_full[0]])
    # the entropy with top seed
    selections.append([[top_seed[0]], top_seed_entropy])
    
    candidates = update_candidates(candidates, selections)
    
    for size in range(2, max_seed_size+1):
        top_seeds = selections[-1][0]
        tmp_new_seeds = []
        tmp_new_seeds_entropy = []
        for candi_seed in candidates:
            tmp_seeds = [tuple(list(top_seeds)+[candi_seed])]
            tmp_module = find_modules(N = N,
                                      seed_size = size,
                                      seed_unit = seed_unit,
                                      _seeds = tmp_seeds,
                                      iterations = iterations,
                                      pinning = pinning,
                                      reduced = reduced,
                                      LUT = LUT,
                                      p = unknown_prob,
                                      data = True,
                                      pin_start = pin_start,
                                      models = models)
            entropy, _ = config_entropy(tmp_module[tmp_seeds[0]], base=2, normalized=True)
            entropy = entropy[iterations]
            tmp_new_seeds.append(tmp_seeds[0])
            tmp_new_seeds_entropy.append(entropy)
        
        new_entropy = min(tmp_new_seeds_entropy)
        new_top_seeds = tmp_new_seeds[tmp_new_seeds_entropy.index(new_entropy)]
        
        selections.append([new_top_seeds, new_entropy])
        candidates = update_candidates(candidates, selections)
    
    return selections