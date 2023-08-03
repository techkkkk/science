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


def get_outlayer_infos(atoms: Atoms, distances, max_z_atom_idx: list, min_z_atom_idx: list, threshold=0.5):
    '''
        threshold:近外层范围R的阈值, R > max_atom_z - threshold.或 R < min_atom_z + threshold
    '''
    # 绝对长度转为相对值
    threshold = threshold/atoms.get_cell_lengths_and_angles()[2]
    electronegativity = {
        "H": 2.20, "He": 0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
        "O": 3.44, "F": 3.98, "Ne": 0, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.98,
        "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0, "K": 0.82, "Ca": 1.00, "Sc": 1.36,
        "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.92,
        "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
        "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.59, "Mo": 2.16,
        "Tc": 1.91, "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78,
        "Sn": 1.96, "Sb": 2.05, "Te": 2.12, "I": 2.66, "Xe": 2.60, "Cs": 0.79, "Ba": 0.89,
        "La": 1.11, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.2,
        "Gd": 1.21, "Tb": 1.13, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.26,
        "Lu": 1.27, "Hf": 1.32, "Ta": 1.51, "W": 2.36, "Re": 1.93, "Os": 2.18, "Ir": 2.20,
        "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 1.87, "Bi": 2.02, "Po": 1.99, "At": 2.22, "Th": 1.3,
        "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.13, "Cm": 1.28, "Bk": 1.3,
        "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3, "Lr": 1.3, "Rf": 1.3, "Db": 1.3,
        "Sg": 1.3, "Bh": 1.3, "Hs": 1.3, "Mt": 1.3
    }

    atomic_radiis = {'H': 0.32, 'He': 0.46, 'Li': 1.33, 'Be': 1.02, 'B': 0.85, 'C': 0.75,
                     'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67, 'Na': 1.55, 'Mg': 1.39,
                     'Al': 1.26, 'Si': 1.16, 'P': 1.11, 'S': 1.03, 'Cl': 0.99, 'Ar': 0.96,
                     'K': 1.96, 'Ca': 1.71, 'Sc': 1.48, 'Ti': 1.36, 'V': 1.34, 'Cr': 1.22,
                     'Mn': 1.19, 'Fe': 1.16, 'Ni': 1.10, 'Co': 1.11, 'Cu': 1.12, 'Zn': 1.18,
                     'Ga': 1.24, 'Ge': 1.24, 'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17,
                     'Rb': 2.10, 'Sr': 1.85, 'Y': 1.63, 'Zr': 1.54, 'Nb': 1.47, 'Mo': 1.38,
                     'Tc': 1.28, 'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.20, 'Ag': 1.28, 'Cd': 1.36,
                     'In': 1.42, 'Sn': 1.40, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
                     'Cs': 2.32, 'Ba': 1.96, 'La': 1.80, 'Ce': 1.63, 'Pr': 1.76, 'Nd': 1.74,
                     'Pm': 1.73, 'Sm': 1.72, 'Eu': 1.68, 'Gd': 1.69, 'Tb': 1.68, 'Dy': 1.67,
                     'Ho': 1.66, 'Er': 1.65, 'Tm': 1.64, 'Yb': 1.70, 'Lu': 1.62, 'Hf': 1.52,
                     'Ta': 1.46, 'W': 1.37, 'Re': 1.31, 'Os': 1.29, 'Ir': 1.22, 'Pt': 1.23,
                     'Au': 1.24, 'Hg': 1.33, 'Tl': 1.44, 'Pb': 1.44, 'Bi': 1.51}
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

    # 统计近外层数目最多原子及其占比，max_atom_z - threshold 和 min_atom_z + threshold
    # 范围内原子合并计数
    atom_count = {}
    near_outlayer_total_cnt = 0
    positions = atoms.get_scaled_positions()
    for i in range(len(atoms)):
        atom = atoms[i]
        if positions[i,2] > positions[max_z_atom_idx[0], 2]-threshold \
            or positions[i,2] < positions[min_z_atom_idx[0], 2]+threshold:
            near_outlayer_total_cnt += 1
            atom_type = atom.symbol
            if atom_type in atom_count:
                atom_count[atom_type] += 1
            else:
                atom_count[atom_type] = 1

    max_count_atom_type = max(atom_count, key=atom_count.get)
    max_atom_count = atom_count[max_count_atom_type]
    res["near_outlayer_infos"]["near_outlayer_total_cnt"] = near_outlayer_total_cnt
    res["near_outlayer_infos"]["max_atom"] = max_count_atom_type
    res["near_outlayer_infos"]["max_atom_count"] = max_atom_count
    res["near_outlayer_infos"]["max_atom_ratio"] = max_atom_count/near_outlayer_total_cnt
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
            near_outer_atoms.append(infos["near_outlayer_infos"]["max_atom"])
            print(infos["near_outlayer_infos"]["max_atom"])  # 近外层占比最大的原子符号
            print(infos["near_outlayer_infos"]["max_atom_count"]) # 近外层占比最大的原子数量
            print(infos["near_outlayer_infos"]["near_outlayer_total_cnt"]) # 近外层原子总数
            print(infos["near_outlayer_infos"]["max_atom_ratio"]) # 近外层占比最大的原子比例
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

