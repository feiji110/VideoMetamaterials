#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   formatdatabase.py
@Time    :   2024/05/01 22:22:35
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
@Desc    :  
将所有文件有序化
使得所有文件和csv文件中的数据一一对应
同时去除index索引，以兼容diffusion的代码

使用脚本前应当：
1. 人工删除有问题的topo文件
2. 修改工作路径
'''

import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--workdir', type=str, default='/home/hengda/VideoMetamaterials/data/validation/', help='work dir')
    return parser.parse_args()



def rename(folder='topo/'):
    """将文件名批量重命名"""
    exts = ['gif']
    topo_folder = f'./gifs/{folder}/'
    paths_top = [p for ext in exts for p in Path(f'{topo_folder}').glob(f'**/*.{ext}')] # 训练样本数，简化为16
    # sort paths by number of name
    paths_top = sorted(paths_top, key=lambda x: int(x.name.split('.')[0]))
    # logging.info(paths_top)
    for i, p in enumerate(paths_top):
        p.rename(f'{topo_folder}{i}.gif')

def reset_index():
    folder = os.listdir('./gifs/')
    for i in folder:
        # removefile(i)
        rename(i)

def get_topo_list():
    """获取topo文件名列表"""
    topo_folder = './gifs/topo/'
    paths_top = list(Path(f'{topo_folder}').glob('**/*gif'))
    # sort paths by number of name
    paths_top = sorted(paths_top, key=lambda x: int(x.name.split('.')[0]))
    return [int(p.name.split('.')[0]) for p in paths_top]

def read_csv():
    """ 
    1. 将lag_frame_ranges.csv 文件 与 stress_strain_data.csv 文件中的数据读取出来，
    2. 根据sub_list(排序好的topo文件名)文件 检查在csv文件中可以读取正确条目
    """
    import pandas as pd
    lag_frame_ranges = pd.read_csv('./ori-lag_frame_ranges.csv', header=None, index_col=0, names=['min_disp_x', 'max_disp_x', 'min_disp_y', 'max_disp_y',  'min_s_22', 'max_s_22'])
    stress_strain_data = pd.read_csv('./ori-stress_strain_data.csv', header=None, index_col=0 )
    # sub_list 保证有序
    sub_list= get_topo_list()
    
    # logging.info(sub_list)
    sub_lag = lag_frame_ranges.loc[sub_list]
    sub_stress = stress_strain_data.loc[sub_list ]
    # logging.info(sub_lag)
    # logging.info(sub_stress)
    sub_lag.to_csv('lag_frame_ranges.csv', index=False, header=True)
    sub_stress.to_csv('stress_strain_data.csv', index=False, header=False)
    os.remove('./ori-lag_frame_ranges.csv')
    os.remove('./ori-stress_strain_data.csv')
    

def main():
    get_topo_list()
    read_csv()
    reset_index()

if __name__ == '__main__':
    workdir = parse_args().workdir
    os.chdir(workdir)
    main()   

