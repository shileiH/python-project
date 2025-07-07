# 作者：hsl
# 2025年06月03日20时44分07秒
# 邮箱：2049279114@qq.com

import os
from time import strftime
from time import gmtime

def scan_dir(path, width):
    list_files = os.listdir(path)
    for file_name in list_files:
        print(' ' * width, file_name)
        new_path = path + '/' + file_name  # 确保子目录下的目录，路径正确
        if os.path.isdir(new_path):  # 判断如果是文件夹
            scan_dir(new_path, width + 4)


def use_stat():
    file_info = os.stat('file2') #拿文件状态信息，inode
    print('size{},uid{},mode{:x},mtime{}'.format(file_info.st_size, file_info.st_uid, file_info.st_mode,
                                                 file_info.st_mtime))
    gm_time=gmtime(file_info.st_mtime)
    #把秒数转为字符串时间
    print(strftime("%Y-%m-%d %H:%M:%S", gm_time))

if __name__ == '__main__':
    scan_dir('.', 0)
    use_stat()