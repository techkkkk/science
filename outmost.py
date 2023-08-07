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

def plot_atoms_count_dist(atoms_list, pic_name):
    # 对最外层原子的元素名称进行统计
    counter_unsort = Counter(atoms_list)
    counter = dict(counter_unsort.most_common())
    # # 保存统计结果到文本文件
    with open(structures_dir+f"_{pic_name}_statistics.txt", "w") as f:
        f.write("Element\tCount\n")
        for element, count in counter.items():
            f.write(f"{element}\t{count}\n")

    # 绘制统计图
    elements, counts = zip(*counter.items())
    plt.bar(elements, counts)
    plt.xlabel("Elements")
    plt.ylabel("Counts")
    plt.xticks(rotation=45)
    plt.title(f"Counts_of_{pic_name}")
    plt.savefig(structures_dir+"_"+f"Counts_of_{pic_name}.png",dpi = 500)


def get_shifted_structure(structure):
    # return the thickness and atom coordinate along the current dimension
    def get_thickness_of_slab(atoms=[0.99, 0.01]):

        def make_it_in_01(atoms=[2.554, -0.255]):
            new_atoms = []
            for atom in atoms:
                atom = atom % 1
                new_atoms.append(atom)
            return new_atoms

        def variance(atoms=[2.554, -0.255]):
            var = 0
            for atom in atoms:
                var += (atom - 0.5) ** 2
            return var / len(atoms)

        # store the variance of each move in a list
        var = []
        for i in range(10):
            new_atoms = [j + i / 10 for j in atoms]
            new_atoms = make_it_in_01(new_atoms)
            var.append(variance(new_atoms))

        # get the index of the move with smallest variance
        index = var.index(min(var))
        new_atoms = [j + index / 10 for j in atoms]
        new_atoms = make_it_in_01(new_atoms)
        # calculate the thickness of the slab along current dimension
        thickness = (max(new_atoms) - min(new_atoms))
        return thickness, new_atoms

    # start to run the function to calculate the thickness of the slab along all 3 axis
    vacuum = []
    new_atoms_matrix = None
    # calculate scaled atom coordinate and the thickness of the slab in 3 dimensions, respectively
    for i in range(0, 3):
        thick, new_atoms = get_thickness_of_slab(structure.get_scaled_positions()[:, i])
        vacuum.append(thick * structure.get_cell()[i, i])
        if new_atoms_matrix is None:
            new_atoms_matrix = np.array(new_atoms)
        else:
            new_atoms_matrix = np.c_[new_atoms_matrix, np.array(new_atoms)]

    return new_atoms_matrix, vacuum

def merge_same_and_to_dict(most_atom_bond_info_list):
    # most_atom_bond_info_list转化格式
        result = {}
        for sublist in most_atom_bond_info_list:
            key = tuple(sublist)
            if key in result:
                result[key]['repeat_num'] += 1
            else:
                result[key] = {'symbol': sublist[0], 'length': sublist[1], 'repeat_num': 1}
        return list(result.values())

def get_near_outlayer_most_atoms_bond_infos(atoms2D, rows):
    '''获取most原子成键信息'''
    distances2D = atoms2D.get_all_distances(mic=True)
    bond_info_list = [] # 存储所有 most 原子的bond信息
    for row in rows:    # row:遍历所有 most 原子
        most_atom_bond_info_list = [] # 存储单个 most 原子的bond信息
        for j in range(distances2D.shape[1]): # col:遍历 所有 原子
            atomic_radii_sum = atomic_radiis[atoms2D[row].symbol] + atomic_radiis[atoms2D[j].symbol]
            if distances2D[row][j] > 0.1 and distances2D[row][j] < atomic_radii_sum * 1.4:
                most_atom_bond_info_list.append([atoms2D[j].symbol, distances2D[row][j]])
        
        bond_info_list.append(merge_same_and_to_dict(most_atom_bond_info_list))
    
    unique_data = [x for i, x in enumerate(bond_info_list) if x not in bond_info_list[:i]] #去重
    return unique_data

def get_outlayer_infos(atoms: Atoms, distances, max_z_atom_idx: list, min_z_atom_idx: list, threshold=0.5):
    '''
        threshold:近外层范围R的阈值, R > max_atom_z - threshold.或 R < min_atom_z + threshold
    '''
    # 绝对长度转为相对值
    threshold = threshold/atoms.get_cell_lengths_and_angles()[2]
    from atomic_electronegativity import electronegativity
    
    res = {"outlayer_infos": [], 'ave_saturation': 0, 
           "near_outlayer_infos":{"threshold":threshold,"near_outlayer_total_cnt":0}}
    
    # 统计最外层原子，饱和度
    target_atom_idx = [max_z_atom_idx[0], min_z_atom_idx[0]]
    for i in target_atom_idx:
        infos = {"outer_atom_symbol": atoms[i].symbol, "outer_bonds": [],
                 "electronegativity": electronegativity[atoms[i].symbol]}
        single_atom_bond_quality = electronegativity[atoms[i].symbol]
        for j in range(distances.shape[1]):
            atomic_radii_sum = atomic_radiis[atoms[i].symbol] + atomic_radiis[atoms[j].symbol]
            if distances[i][j] > 0.1 and distances[i][j] < atomic_radii_sum * 1.4:
                outer_bonds = {"connected_atom_symbol": atoms[j].symbol, "atomic_radii_sum": atomic_radii_sum,
                               "real_distance": distances[i][j]}
                #             outer_bonds['single_bond_quality'] = 0.5 * math.exp(3*(atomic_radii_sum - distances[i][j])/atomic_radii_sum)
                outer_bonds['single_bond_quality'] = 0.5 * (atomic_radii_sum / distances[i][j]) ** 4
                single_atom_bond_quality += outer_bonds['single_bond_quality']
                infos['outer_bonds'].append(outer_bonds)
        if len(infos['outer_bonds']) > 0:
            infos['single_atom_bond_quality'] = single_atom_bond_quality
            res['outlayer_infos'].append(infos)
            res['ave_saturation'] += single_atom_bond_quality
        else:
            return None
    res['ave_saturation'] /= len(target_atom_idx)

    # 统计近外层数目最多原子及其占比，max_atom_z - threshold 和 min_atom_z + threshold范围内原子合并计数
    atom_count = {} # 近外层原子种类计数
    atom_idx = {}   # 记录近外层原子index
    near_outlayer_total_cnt = 0
    distances = atoms.get_all_distances(mic=True)
    positions = atoms.get_scaled_positions()
    for i in range(len(atoms)):
        atom = atoms[i]
        if positions[i,2] > positions[max_z_atom_idx[0], 2]-threshold \
            or positions[i,2] < positions[min_z_atom_idx[0], 2]+threshold:
            near_outlayer_total_cnt += 1
            atom_type = atom.symbol
            if atom_type in atom_idx:
                atom_idx[atom_type].append(i)
            else:
                atom_idx[atom_type] = [i]
            if atom_type in atom_count:
                atom_count[atom_type] += 1
            else:
                atom_count[atom_type] = 1


    max_value = max(atom_count.values())
    max_count_atom_type = [atom for atom, count in atom_count.items() if count == max_value]
    max_atom_count = [atom_count[atom_type] for atom_type in max_count_atom_type]
    
    #获取近外层占比最多原子的成键信息
    max_atom_bond_infos = []
    for t in max_count_atom_type: # 有多种most原子时
        bond_info = get_near_outlayer_most_atoms_bond_infos(atoms, atom_idx[t])
        max_atom_bond_infos.append(bond_info)
        
    res["near_outlayer_infos"]["near_outlayer_total_cnt"] = near_outlayer_total_cnt
    res["near_outlayer_infos"]["max_atom"] = max_count_atom_type
    res["near_outlayer_infos"]["max_atom_count"] = max_atom_count
    res["near_outlayer_infos"]["max_atom_ratio"] = [max_atom_count[i]/near_outlayer_total_cnt for i in range(len(max_atom_count))]
    res["near_outlayer_infos"]["max_atom_bond_infos"] = max_atom_bond_infos
    return res


def get_information(poscar_file = "POSCAR1"):
    try:

        atoms = read(poscar_file, format="vasp")

        new_atoms_matrix, vacuum = get_shifted_structure(atoms)
        atoms.set_scaled_positions(new_atoms_matrix)

        distances = atoms.get_all_distances(mic=True)  # 使用周期性边界条件计算距离

        # 获取z轴的最外层原子
        max_z_atom_idx = [np.argmax(atoms.get_scaled_positions()[:, 2])]
        min_z_atom_idx = [np.argmin(atoms.get_scaled_positions()[:, 2])]

        outlayer_infos = get_outlayer_infos(atoms, distances, max_z_atom_idx, min_z_atom_idx, threshold=0.1)

        return outlayer_infos
    
    except Exception as e:
        print("\033[91m{} \033[00m".format(f"ERROR:  reading {poscar_file}: {e}"))
        with open(poscar_file, "r") as f:
            content = f.read()
        print(f"Content of {poscar_file}:\n{content}\n")
        return None


if __name__ == "__main__":
    half_cnt = 0
    outer_atoms = []  # 统计最外层原子的元素名称
    near_outer_atoms = []  # 统计近外层占比最多的原子名称
    error_cnt = 0
    all_infos = {}  # 保存所有信息
    structures_dir = "2D_structure"
    # 读取structures文件夹下的所有POSCAR文件
    poscar_files = glob.glob(os.path.join(structures_dir, "POSCAR*"))  
      
    for f in poscar_files:
        print(f"开始处理{f}")
        infos = get_information(f)
        if infos:
            outer_atoms.append(infos["outlayer_infos"][0]['outer_atom_symbol'])
            outer_atoms.append(infos["outlayer_infos"][1]['outer_atom_symbol'])
            print(infos["outlayer_infos"][0]['outer_atom_symbol'])
            print(infos["outlayer_infos"][1]['outer_atom_symbol'])
            
            # 近外层占比最大的原子信息
            near_outer_atoms.append(infos["near_outlayer_infos"]["max_atom"][0])
            print(infos["near_outlayer_infos"]["max_atom"][0])  # 近外层占比最大的原子符号
            print(infos["near_outlayer_infos"]["max_atom_count"][0]) # 近外层占比最大的原子数量
            print(infos["near_outlayer_infos"]["near_outlayer_total_cnt"]) # 近外层原子总数
            print(infos["near_outlayer_infos"]["max_atom_ratio"][0]) # 近外层占比最大的原子比例
            print(infos["near_outlayer_infos"]["max_atom_bond_infos"][0]) # 近外层占比最大的原子成键信息
            if infos["near_outlayer_infos"]["max_atom_ratio"]==0.5:
                half_cnt += 1
            all_infos[f] = infos
        else:
            error_cnt += 1
        print(f"结束处理{f}\n")

    print(f"finished, total cnt: {len(poscar_files)}, error cnt: {error_cnt}")
    print(f"占比0.5 cnt: {half_cnt}")
    
    #保存所有信息
    all_infos_json = json.dumps(all_infos, indent=4)
    with open(structures_dir+"_"+'all_infos.json', 'w') as f:
        f.write(all_infos_json)    
        
    # 绘图，最外层，近外层原子数    
    plot_atoms_count_dist(outer_atoms, "Outer_Layer_Atoms")
    plot_atoms_count_dist(near_outer_atoms, "Near_Outer_Layer_Atoms")

