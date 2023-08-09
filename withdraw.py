import os
import shutil

target_directory = "marked_structure"  # 目标目录，用于存放移动后的文件

def clear_dir(directory):
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
clear_dir(target_directory)


with open("mark_result.txt") as f:
    lines = f.readlines()

file_names = []

for line in lines:
    if line.strip(): # 检查行是否为空
        columns = line.split() # 将每行内容按空格分割成列
        file_names.append(os.path.basename(columns[0])) # 提取第一列


for filename in file_names:

    target_path = os.path.join(target_directory, filename+"_2D")
    shutil.copy(os.path.join("2D_structure", filename), target_path)

    target_path = os.path.join(target_directory, filename+"_3D")
    shutil.copy(os.path.join("3D_structure", filename), target_path)