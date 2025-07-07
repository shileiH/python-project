import my_print_module


# 另一个py文件里的my_print_module模块
def say_hello():
    print("hello 1")
    print("hello 2")

def sum_num(num1, num2):
    result = num1 + num2
    print("%d + %d = %d" % (num1, num2, result))

def use_print():
    name = "李华"
    student_no = 10001
    print("我的名字叫 %s，请多多关照！" % name)
    print("我的学号是 %06d" % student_no)

# 有7个整数，其中有3个数出现了两次，1个数出现了一次， 找出出现了一次的那个数。
# 异或一个数与自身异或为0，故该7个数异或后剩余的那个就是出现了一次的那个数

def find_single(num):
    result = 0
    for i in num:
        result = result ^ i
    return result


list1 = [1, 2, 3, 3, 1, 2, 4]
print(find_single(list1))


def printf():
    for i in range(1, 21):
        print(i, end=" ")
    print()


num = 5  # "具体打印几次依靠传递的参数num"


def say_hel(n) -> num:
    '''
    打印num个hello
    :param n:
    '''
    for i in range(1, n + 1):
        print("hello")


printf()
say_hel(num)
my_print_module.say_hello()
my_print_module.sum_num(3, 6)
my_print_module.use_print()


def find_two(num):
    # 第一步整体异或得到两个数的异或结果mask
    mask = 0
    result1 = result2 = 0
    for i in num:
        mask = mask ^ i   #按位异或1，0和0，1为1
    mask = mask & (-mask)  #按位与只有1与1才得1
    print(mask)

    for j in num:
        if j & mask != 0:
            result1 = result1 ^ j
        else:
            result2 = result2 ^ j
    return [result1, result2]


list1 = [1, 2, 3, 3, 1, 2, 4, 5]
print(find_two(list1))
