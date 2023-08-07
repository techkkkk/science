import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
from ase.io import read
from ase import Atoms, Atom
from ase.geometry import get_layers
import numpy as np
import os
import math
import json
import pprint
from atomic_radiis import atomic_radiis
from copy import deepcopy

def is_same_pos(x, y):
    return (x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2 < 0.01

def mark_is_ok(f3D, near_outlayer_infos, thred1=0.01, thred2=0.1):
    """是否需要标记

    Args:
        f3D (_type_): 3D file
        near_outlayer_infos (_type_): _description_
        thred1 (float): 判断2D 3D键长是否相同的长度误差
        thred2 (float): 判断匹配之后，剩余 3D键长是否与已匹配键长处于相同长度范围的阈值
    """
    atoms3D = read(f3D, format="vasp")
    distances3D = atoms3D.get_all_distances(mic=True)
    
    from atomic_radiis import atomic_radiis
    
    bond_infos_in2D = near_outlayer_infos["max_atom_bond_infos"][0]
            
    max_atom = near_outlayer_infos["max_atom"][0]
    # print(bond_infos_in2D)    
    for i in range(len(atoms3D)): # 遍历所有 3D 原子
        if atoms3D[i].symbol == max_atom:   # 与2D最外层元素相同的 3D 原子
            atoms3D_bonds = []
            for j in range(distances3D.shape[1]): # 记录3D原子的成键信息
                atomic_radii_sum = atomic_radiis[atoms3D[i].symbol] + atomic_radiis[atoms3D[j].symbol]
                if distances3D[i][j] > 0.1 and distances3D[i][j] < atomic_radii_sum * 1.4:
                    atoms3D_bonds.append( [atoms3D[j].symbol, distances3D[i][j] ] )
            
            # 与 2D 的成键信息比对
            for bonds in bond_infos_in2D: # 遍历所有 2D 最外层原子成键信息
                bonds_2D = deepcopy(bonds)
                bonds_3D = deepcopy(atoms3D_bonds)
                bonds_3D_matched_indices = []
                for i, bond3D in enumerate(bonds_3D): # 遍历 3D 原子的每一个键
                    for bond2D in bonds_2D: # 遍历单个 2D 最外层原子的成键键信息
                        if bond3D[0] == bond2D['symbol'] and abs(bond3D[1]-bond2D['length'])<thred1 \
                        and bond2D['repeat_num'] > 0:
                            bonds_3D_matched_indices.append(i)
                            bond2D['repeat_num'] -= 1
                            
                # 匹配完后检查是否匹配成功(repeat_num==0)，是否需要mark(bonds_3D剩余有类似键长)：
                all_matched = True
                for bond2D in bonds_2D: # 检查是否匹配成功
                    if bond2D['repeat_num'] > 0:
                        all_matched = False
                        break
                if all_matched: # 如果匹配成功，检查需要mark(bonds_3D剩余有类似键长)：
                    
                    for i, bond3D in enumerate(bonds_3D):
                        if i not in bonds_3D_matched_indices: # 检查未匹配的键长：
                            if bond3D[0] == bond2D['symbol'] and abs(bond3D[1]-bond2D['length'])<thred2:
                                print(f"should mark {bond3D[0]}")
                                return True
    return False
    
def get_near_outlayer_infos(poscar_file, info_file="2D_structure_all_infos.json"):
    '''读取近外层信息'''
    if not hasattr(get_near_outlayer_infos, "all_infos_dict"):
        with open(info_file) as f:
            get_near_outlayer_infos.all_infos_dict = json.load(f)
    if poscar_file in get_near_outlayer_infos.all_infos_dict.keys():
        return get_near_outlayer_infos.all_infos_dict[poscar_file]["near_outlayer_infos"]
    else:
        return None
    
if __name__ == "__main__":
    
    error_cnt = 0
    near_outer_atoms_infos_dict = {}  # 保存所有信息

    # 读取structures文件夹下的所有POSCAR文件
    poscar_files_2D = glob.glob(os.path.join("2D_structure", "POSCAR*"))  
    poscar_files_3D = glob.glob(os.path.join("3D_structure", "POSCAR*"))  
      
    for i, f2D in enumerate(poscar_files_2D):
        f3D = poscar_files_3D[i]
        print(f"[{i}] 开始处理{f2D}")
        near_outlayer_infos = get_near_outlayer_infos(f2D)
        if near_outlayer_infos:
            mark_is_ok(f3D, near_outlayer_infos)
        else:
            error_cnt += 1
        print(f"[{i}] 结束处理{f2D}\n")

    print(f"finished, total cnt: {len(poscar_files_2D)}, error cnt: {error_cnt}")


