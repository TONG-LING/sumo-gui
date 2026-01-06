#!/usr/bin/env python3
"""
创建程序运行所需的所有目录
"""
import os
import config

def create_directories():
    """创建所有必要的目录"""
    directories = [
        config.DATA_DIR,
        config.WEIGHTS_DIR,
        config.CPM_DIR,
        config.SAMPLES_DIR,
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
        else:
            print(f"目录已存在: {directory}")

if __name__ == "__main__":
    create_directories()
    print("所有必要目录已创建完成!")
