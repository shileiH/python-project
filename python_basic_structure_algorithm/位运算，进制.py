a = 5
b = 6.5
c = True
print(f"a类型为{type(a)}\nb类型为{type(b)}\nc类型为{type(c)}")
y = complex(2, 1)
print("y类型为%s" % type(y))
lista = [1, 2, 3]
tuple1 = (2, 5, 7)
dictx = {"name": "王五", "age": 18}
print("dictx类型为", type(dictx))

# 输出一个整数的二进制，八进制，十六进制
a = int(input("请输入一个整数："))
print("整数a的二进制为%s，八进制为%s，十六进制为%s" % (bin(a), oct(a), hex(a)))


def sum_odds():
    sum = 0
    for i in range(1, 100):
        if i % 2 == 1:
            sum += i
    print(sum)


sum_odds()


def mul_table():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j} * {i} = {i * j}\t", end="")  # \t为制表符
        print()  # 换行


mul_table()


# 统计一个整数对应的二进制数的1的个数
# 使用位运算
def count_ones(num):
    check_flag = 1
    count = 0  # 为1的数目
    i = 1
    while i <= 32:  # 32位，64位最好
        if check_flag & num:  # 判断某一位是否为1
            count += 1
        check_flag = check_flag << 1  # 不断的往左移动*2
        i += 1
    print(count)


num = int(input('请输入1个数：'))
count_ones(num)
n = num
