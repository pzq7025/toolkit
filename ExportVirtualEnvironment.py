import os
import subprocess
import sys

# 使用Python导出所有虚拟环境的配置，由文心一言3.5提供源码。只修改了父级目录，其他没做任何修改。

# 虚拟环境所在的父目录
venv_parent_dir = r'E:\Drive\\virtualEnv'  # 替换为你的虚拟环境父目录

# 遍历虚拟环境目录
for venv_dir in os.listdir(venv_parent_dir):
    venv_path = os.path.join(venv_parent_dir, venv_dir)
    if os.path.isdir(venv_path):
        # 检查是否为虚拟环境（例如，检查是否存在bin/python或Scripts/python.exe）
        python_bin = os.path.join(venv_path, 'bin', 'python') if sys.platform != 'win32' else os.path.join(venv_path, 'Scripts', 'python.exe')
        if os.path.exists(python_bin):
            # 获取Python版本
            python_version_cmd = [python_bin, '--version']
            python_version_result = subprocess.run(python_version_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            python_version = python_version_result.stdout.strip() if python_version_result.returncode == 0 else 'Unknown'

            # 获取pip包列表
            pip_list_cmd = [os.path.join(venv_path, 'bin', 'pip') if sys.platform != 'win32' else os.path.join(venv_path, 'Scripts', 'pip.exe'), 'list', '--format=freeze']
            pip_list_result = subprocess.run(pip_list_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pip_packages = pip_list_result.stdout.strip() if pip_list_result.returncode == 0 else 'Error retrieving package list'
            print(pip_list_cmd)
            print(f'{"=" * 50}{"*" * 20}{"=" * 50}')
            # 创建txt文件并写入信息
            output_file = os.path.join(venv_path, 'config.txt')
            with open(output_file, 'w') as f:
                f.write(f"Virtual Environment: {venv_dir}\n")
                f.write(f"Python Version: {python_version}\n\n")
                f.write("Installed Packages:\n")
                f.write(pip_packages)
            print(f"Exported config for {venv_dir} to {output_file}")
        else:
            print(f"Skipping {venv_dir} as it does not seem to be a virtual environment.")
