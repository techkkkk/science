import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
from ase.io import read
from ase import Atoms, Atom
from ase.geometry import get_layers
import numpy as np
import os
import matplotlib.pyplot as plt
import time
# import plotly.distances_objects as go


class Util():
    rvdws_dict = None
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
                    'Au': 1.24, 'Hg': 1.33, 'Tl': 1.44, 'Pb': 1.44, 'Bi': 1.51 }    
    
    @classmethod
    def get_rvdws(cls):
        '''范德瓦尔斯半径'''
        rvdws = """
        1 H 1.20
        2 He 1.43
        3 Li 2.12 76 11 067 Zero-coordinated Li disregarded
        4 Be 1.98 90 3515
        5 B 1.91 70 152 194
        6 C 1.7 1.77 82 385 475 Only sp3 C atoms, R ≤ 2.5%
        7 N 1.66 52 187 967 N⋯N contacts, R ≤ 3.5%
        8 O 1.50 73 420 207 OH excluded, R ≤ 3.5%
        9 F 1.46 66 497 497 OH excluded
        10 Ne 1.58 12 Ne⋯Ne in solid Ne
        11 Na 2.50 51 16 016 Zero-coordinated Na disregarded, OH excluded
        12 Mg 2.51 92 11 581 Zero-coordinated Mg disregarded
        13 Al 2.25 75 9877
        14 Si 2.19 55 15 077
        15 P 1.90 67 178 077 OH excluded
        16 S 1.89 58 741 158 OH excluded, R ≤ 7.5%
        17 Cl 1.82 69 641 448 OH excluded, monocoordinated Cl only
        18 Ar 1.83 93 527 Ar⋯C contacts
        19 K 2.73 28 76 013 Zero-coordinated K disregarded
        20 Ca 2.62 78 5420 Zero-coordinated Ca disregarded
        21 Sc 2.58 93 1287
        22 Ti 2.46 36 6685
        23 V 2.42 37 17 485 Polyoxometallates and OH excluded
        24 Cr 2.45 71 60 314
        25 Mn 2.45 79 81 976
        26 Fe 2.44 67 207 868
        27 Co 2.40 81 186 046
        28 Ni 2.40 76 115 164
        29 Cu 2.38 75 42 451 Only six-coordinated Cu
        30 Zn 2.39 74 68 186
        31 Ga 2.32 80 6066
        32 Ge 2.29 50 13 207
        33 As 1.88 54 22 962 OH excluded
        34 Se 1.82 46 36 624 OH excluded
        35 Br 1.86 68 172 324 Monocoordinated Br only
        36 Kr 2.25 98 131
        37 Rb 3.21 33 1960 Zero-coordinated Rb disregarded
        38 Sr 2.84 84 2094 Zero-coordinated Sr disregarded
        39 Y 2.75 89 3487
        40 Zr 2.52 64 5523
        41 Nb 2.56 50 3647 Polynuclear excluded
        42 Mo 2.45 63 138 249
        43 Tc 2.44 79 2880
        44 Ru 2.46 73 165 471
        45 Rh 2.44 54 34 854
        46 Pd 2.15 49 35 830
        47 Ag 2.53 50 27 221 Zero-coordinated Ag disregarded
        48 Cd 2.49 74 21 952
        49 In 2.43 80 5230
        50 Sn 2.42 44 30 075
        51 Sb 2.47 26 15 850 Polynuclear excluded
        52 Te 1.99 67 13 772 OH excluded
        53 I 2.04 53 56 317 Monocoordinated I only
        54 Xe 2.06 86 2264 Xe⋯C contacts
        55 Cs 3.48 46 775 Zero-coordinated Cs disregarded
        56 Ba 3.03 58 2402 Zero-coordinated Ba disregarded
        57 La 2.98 80 6471 Zero-coordinated La disregarded
        58 Ce 2.88 81 3681 Zero-coordinated Ce disregarded
        59 Pr 2.92 78 3360
        60 Nd 2.95 81 6346
        62 Sm 2.90 84 4162
        63 Eu 2.87 80 7042
        64 Gd 2.83 84 6682
        65 Tb 2.79 81 5538
        66 Dy 2.87 83 3615
        67 Ho 2.81 82 2493
        68 Er 2.83 86 4246
        69 Tm 2.79 84 1141
        70 Yb 2.80 83 4664
        71 Lu 2.74 86 2018
        72 Hf 2.63 86 936
        73 Ta 2.53 80 2793 Polynuclear excluded
        74 W 2.57 62 47 936 Polynuclear excluded
        75 Re 2.49 75 61 593
        76 Os 2.48 90 129 040
        77 Ir 2.41 73 18 335
        78 Pt 2.29 67 55 873
        79 Au 2.32 49 5132 Only square planar Au
        80 Hg 2.45 81 30 628 Coordination number two or higher for Hg
        81 Tl 2.47 71 3486
        82 Pb 2.60 49 36 781
        83 Bi 2.54 43 14 030
        89 Ac 2.8 33 Ac⋯Cl contacts in AcCl3
        90 Th 2.93 83 964
        91 Pa 2.88 94 48 Pa⋯C contacts in (NEt4)[PaOCl5]
        92 U 2.71 74 35 070 OH excluded
        93 Np 2.82 55 830 OH excluded
        94 Pu 2.81 77 1299
        95 Am 2.83 94 128
        96 Cm 3.05 99 90
        97 Bk 3.4 3 Bk⋯Cl contacts in Cs2BkCl6
        98 Cf 3.05 100 14 Only one crystal structure
        99 Es 2.7 2 Es⋯Cl contacts in EsCl3
        """
        if cls.rvdws_dict is None:
            cls.rvdws_dict = {row[1]: float(row[2]) for row in [l.strip().split() for l in rvdws.splitlines() if len(l) > 0] if len(row)>0}
        return cls.rvdws_dict

    @classmethod
    def get_atomic_radiis(cls):
        '''原子半径'''
        return cls.atomic_radiis

    @classmethod
    def make_dir(cls, directory):
        # 判断目录是否存在
        if not os.path.exists(directory):
            # 如果目录不存在，则创建目录
            os.makedirs(directory)
        else:
            # 如果目录存在，则删除目录下的所有文件和子目录
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)

    # @classmethod
    # def plot_groups(cls, atoms:Atoms, groups, connect_infos, output='group_plot.html'):
    #     group_line_color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    #         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    #         'crimson', 'limegreen', 'dodgerblue', 'darkviolet', 'gold',]
    #     atoms_color_list = ['Viridis','Reds','Blues','Jet','YlOrRd','Rainbow','Greens']
        
    #     symbol_list = list(set(atoms.get_chemical_symbols()))
    #     from collections import defaultdict
    #     atom_groups = defaultdict(list)

    #     # 将原子按元素符号分组
    #     for id in range(len(atoms)):
    #         atom_groups[atoms[id].symbol].append([atoms[id], id])

    #     fig = go.Figure(data=go.Scatter3d())
    #     # 打印每个分组中的原子
    #     for symbol, group in atom_groups.items():
    #         color = atoms_color_list[symbol_list.index(symbol)]
    #         for atom_info in group:
    #             atom, id = atom_info[0],atom_info[1] 
    #             u = np.linspace(0, 2 * np.pi, 100)
    #             v = np.linspace(0, np.pi, 50)
    #             x_center, y_center, z_center = atom.position[0], atom.position[1], atom.position[2]
    #             x = x_center + cls.get_rvdws()[atom.symbol]/3 * np.outer(np.cos(u), np.sin(v))
    #             y = y_center + cls.get_rvdws()[atom.symbol]/3 * np.outer(np.sin(u), np.sin(v))
    #             z = z_center + cls.get_rvdws()[atom.symbol]/3 * np.outer(np.ones(np.size(u)), np.cos(v))
    #             # 添加球体到图形对象中
    #             title = f'{str(id)}:{str(atom.symbol)}:[{x_center}  {y_center}  {z_center}]'
    #             fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=color, showscale=False,
    #                 name=title,
    #                 legendgroup=color, showlegend=True))

    #     for i in range(len(connect_infos)):
    #         for j in range(len(connect_infos[0])):
    #             if connect_infos[i][j] == 1:
    #                 connection_data = go.Scatter3d(
    #                     x=[atoms[i].position[0], atoms[j].position[0]],
    #                     y=[atoms[i].position[1], atoms[j].position[1]],
    #                     z=[atoms[i].position[2], atoms[j].position[2]],
    #                     mode='lines',
    #                     line=dict(color="black", width=10),
    #                     showlegend=False
    #                 )
    #                 fig.add_trace(connection_data)

    #     layout = go.Layout(
    #         scene=dict(
    #             xaxis=dict(title='X'),
    #             yaxis=dict(title='Y'),
    #             zaxis=dict(title='Z')
    #         ),
    #         showlegend=True
    #     )
    #     fig.layout = layout
    #     fig.write_html(output)
    


   
class GroupSplit():
    def __init__(self,) -> None:
        self.total_cnt, self.error_cnt, self.ok_cnt = 0, 0, 0

    def set_input_dir(self, input_dir):
        self.poscar_files = glob.glob(os.path.join(input_dir, "POSCAR*"))        
        return self.poscar_files
        
    def set_output_dir(self, out_dir):
        self.out_dir = out_dir
        Util.make_dir(out_dir)


    def split_groups(self, extend_atoms:Atoms, limit_ratio=1, delta=0):
        '''
            按范德瓦尔斯距离分组
                判定同组条件:
                    物理距离( atomA, atomB ) < limit_ratio * ( 范德瓦尔斯距离( atomA, atomB )) - delta
        '''
        # 不使用周期性边界条件计算所有原子距离
        distances = extend_atoms.get_all_distances(mic=False)
        
        connect_infos = [[0] * len(extend_atoms) for _ in range(len(extend_atoms))]
        
        def dfs(distances, visited, cur, same_group_index):
            visited[cur] = True
            same_group_index.append(cur)
            for i in range(len(distances)):
                dis = limit_ratio*abs(Util.get_rvdws()[extend_atoms[cur].symbol]+Util.get_rvdws()[extend_atoms[i].symbol])-delta
                if not visited[i] and distances[cur][i] < dis:
                    connect_infos[cur][i] = connect_infos[i][cur] = 1
                    dfs(distances, visited, i, same_group_index)
                    
        n = len(distances)
        visited = [False] * n
        result = []
        for i in range(n):
            if not visited[i]:
                same_group_index = []
                dfs(distances, visited, i, same_group_index)
                result.append(same_group_index)
        return result



    def do_split(self, idx, poscar_file, extend_num, is_use_record_delta, is_print_details, plot_output="plot.html"):
        '''
            @extend_num: 扩胞数
            @is_use_record_delta: 是否跳过搜索过程使用缓存的delta，
                        搜索过程:
                            最大分组原子数：所有分组的原子数中的最大值
                            有效分组: 组内原子数超过 最大分组原子数的 1/3
                            搜索delta(分组判断依据)过程: 有效分组数 < 3时 delta+=0.1  如果delta>4 分组失败
            @is_print_details: 是否打印搜索过程
        '''
        try:
            atoms = read(poscar_file, format="vasp")
            extend_atoms = atoms * (extend_num, extend_num, extend_num)

            delta = 0
            if is_use_record_delta:
                is_ok = 1
                while True:
                    groups = self.split_groups(extend_atoms, 1, delta)

                    # 获取有效分组数 count
                    group_num = len(groups)
                    max_len = max(len(sub_arr) for sub_arr in groups)
                    count = sum(1 for sub_arr in groups if len(sub_arr) > max_len / 3)

                    # print(f"[try][{idx}][{poscar_file}] group_num[{group_num}] delta[{delta}] ]")
                    if is_print_details:
                        for g in groups:
                                print(len(g), end=' ')
                        print(f"delta={delta} group_num={group_num} count={count}")
                        print('\n')
                    # 有效分组数 count < 3，继续增加delta
                    if count <3:
                        delta += 0.1
                    else:
                        break
                    if delta > 4:
                        is_ok = 0
                        print("\033[91m{} \033[00m".format(f"No proper group found:  reading {poscar_file}: {e}"))
                        break
                content = f"{idx} {is_ok} {poscar_file} {group_num} {count} {delta}"
                for g in groups:
                    content += " " + str(len(g))
                
                with open(f'{self.out_dir}/delta_{os.path.basename(poscar_file)}.txt', 'w') as delta_f:
                    delta_f.write(content)
            else:
                with open(f'{self.out_dir}/delta_{os.path.basename(poscar_file)}.txt', 'r') as delta_f:
                    delta_string = delta_f.read()
                row = delta_string.strip().split()
                is_ok = int(row[1])
                if not is_ok:
                    print("\033[91m{} \033[00m".format(f"No proper group found:  reading {poscar_file}: {e}"))
                    return 0
                record_delta = float(row[5])
                record_count = int(row[4])
                record_group_num = int(row[3])

                groups = self.split_groups(extend_atoms, 1, record_delta)
                # plot_groups(extend_atoms,groups,connect_infos, plot_output)
                # print(groups)
                group_num = len(groups)
                max_len = max(len(sub_arr) for sub_arr in groups)
                count = sum(1 for sub_arr in groups if len(sub_arr) > max_len / 3)
                assert(record_group_num == group_num)
                assert(count == record_count)
            
            group_lens = []
            for g in groups:
                group_lens.append(len(g))
            group_lens.sort(reverse=True)
            print(group_lens)
            print(f"[{idx}][{poscar_file}] group_num[{group_num}] count[{count}] delta[{delta}] ]")
            # plot_groups(extend_atoms,groups,connect_infos, plot_output)
            # for g in groups:
            #     print(len(g), end=' ')
            print('\n')
            return 0
        except Exception as e:
            print("\033[91m{} \033[00m".format(f"ERROR:  reading {poscar_file}: {e}"))
            with open(poscar_file, "r") as f:
                content = f.read()
            print(f"Content of {poscar_file}:\n{content}\n") 
            return 1


    def run_tasks(self, poscar_files):
        start_time = time.time()
        
        for idx, poscar_file in enumerate(poscar_files):
            if idx ==0 or idx >5:
                continue
            res = self.do_split(idx, poscar_file, extend_num=3, is_use_record_delta=True, is_print_details=True, plot_output='plot1.html')

            if res == 1:
                self.error_cnt += 1
            elif res == 0:
                self.ok_cnt += 1

        print(f"ok cnt: {self.ok_cnt}, error cnt: {self.error_cnt}")
        end_time = time.time()
        run_time = end_time - start_time
        print("程序运行时间：", run_time, "秒")

if __name__ == "__main__":
    try:
        g = GroupSplit()
        poscar_files = g.set_input_dir('retbulk')
        g.set_output_dir('retbulk_delta')
        
        g.run_tasks(poscar_files)
        
    except Exception as e:
        print(e)