import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建到另一个文件夹中文件的相对路径
target_path = os.path.join(current_dir, "..", "data", "aa")
target_path=os.path.normpath(target_path)
print(current_dir)
print(target_path)
print()