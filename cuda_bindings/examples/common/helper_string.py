# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys
import os
import logging

def checkCmdLineFlag(stringRef):
    return any(stringRef == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))


def getCmdLineArgumentInt(stringRef):
    for i, k in enumerate(sys.argv):
        if stringRef == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0


def sdkFindFilePath(filename, executable_path):
    """ 从executable_path的目录中查找文件的路径
    
    :filename: 要查找的文件名
    :executable_path: 可执行文件的路径
    
    """

    platform = sys.platform
    if platform.startswith("win32") :
        delimiter_pos = executable_path.rfind("\\")
        executable_dir = executable_path[:delimiter_pos + 1] if delimiter_pos != -1 else str

    else:
        delimiter_pos = executable_path.rfind("/")
        executable_dir = executable_path[:delimiter_pos + 1] if delimiter_pos != -1 else str
    
    path = executable_dir + filename
    try:
        print(os.path.abspath(path))
        with open(path, 'r') as f:
            return path
    except FileNotFoundError:
        pass

    return None
