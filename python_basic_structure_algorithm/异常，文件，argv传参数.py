# 作者：hsl
# 2025年06月03日17时57分09秒
# 邮箱：2049279114@qq.com
import sys
import os

# 1、通过try进行异常捕捉，确保输入的内容一定是一个整型数，然后判断该整型数是否是对称数，12321就是对称数，
# 123321也是对称数，否则就打印不是，非整型抛异常，不是对称数抛异常
#
# 2、传递参数file1，通过sys.argv[1]打开文件，读取里边的内容并打印，如果传递的参数是file2，程序同样可以打印file2的文件内容
#
# 3.新建包，并使用自己建的包
# 4.完成目录深度优先遍历
# 5.完成栈的编写


def mirror():
    num = int(input("请输入一个整型数："))
    x = 0
    y = num
    while y >= 1:
        x = x * 10 + y % 10
        y = y // 10
    if x == num:
        print(f'{num}是对称数')
    else:
        print(f'不是对称数{num}抛异常')

try:
    mirror()
except  Exception as e: #ValueError错误
    print(f'非整型抛异常')
    print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
    print(e.__traceback__.tb_lineno)  # 发生异常所在的文件行号

# os.rename('file','file1')
print(sys.argv)
# print(sys.argv[1])
file = open(sys.argv[1], encoding='utf8')
print(file.read())
file.close()

def copy_file():
    file_read = open("file1")
    file_write = open("file2", "w")
    # 2. 读取并写入文件
    text = file_read.read()
    file_write.write(text)
    # 3. 关闭文件
    file_read.close()
    file_write.close()


def file_open_a2():
    copy_file()
    text1 = open('file2', mode='a+', encoding='utf-8')
    text1.write("wangdao")
    print(text1.read())
    text1.close()

if __name__ == '__main__':
    file_open_a2()
    file = open(sys.argv[2], encoding='utf8')
    print(file.read())
    file.close()
