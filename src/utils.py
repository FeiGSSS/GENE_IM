# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/08/25 09:59:47
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

from cana.utils import statenum_to_binstate
from cana.boolean_node import BooleanNode
from cana.boolean_network import BooleanNetwork

from typing import Dict

def look_up_table(node:BooleanNode):
    """构建节点的输入和输出的映射 
    输入用一个01序列表示
    输出为0或者1
    """
    d = {}
    for statenum, output in zip(range(2**node.k), node.outputs):
        # Binary State, Transition
        inputs = statenum_to_binstate(statenum, base=node.k)
        d[inputs] = output
    return d

def LUT_reduce(d:Dict)->Dict:
    """对于binary的mapping，输入只会导致两个结果，0或者1
    那么只需要保留数目较少的那个结果，就可以简化这个mapping。
    """
    outs = list(d.values())
    assert set(outs) == set(['0','1'])
    zero_cnt = outs.count('0')
    one_cnt  = outs.count('1')
    
    reduced_d = {}
    if zero_cnt != 0 and one_cnt != 0:
        minor_state = '0' if zero_cnt < one_cnt else '1'
        reduced_d = {k:v for k,v in d.items() if v==minor_state}
    
    return reduced_d

def get_sunits(N:BooleanNetwork):
    """为每个基因生成两种状态对应的节点
    例如[A,B,C]这三个基因，生成[A-0,A-1,B-0,B-1,C-0,C-1]这样的序列。
    """
    sunit = []
    for node in N.nodes:
        for state in ['0','1']:
            sunit.append(str(node.name)+'-'+state)
    return sunit