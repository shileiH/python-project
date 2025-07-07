# 作者：hsl
# 2025年06月03日17时54分09秒
# 邮箱：2049279114@qq.com

# 列表
# 增加（three ways)
list1 = ["李华", "小明", "xiaozhang", "wangwu"]
list1.append("王五")
print(list1)
list1.insert(2, "小李")
print(list1)
# extend可追加另一个列表
list2 = [3, 2, 1]
list2.extend(list1)
print(list2)

# 删除4 del在前，pop末尾，remove指定value，clear全部
del list1[0]
list1.pop()
list1.remove("小明")
print(list1)
list2.clear()
print(list2)

# 查找len,count,index首次
len(list1)
list1.append("xiaozhang")
list1.count("xiaozhang")
list1.index("xiaozhang")

# 修改
list1[0] = "1"

# 字典
dict1 = {"name": "小明",
         "age": 18,
         "gender": True,
         "height": 1.75}

# 增加2 key=value,setdefault(key,value)
dict1["friend"] = "李华"
dict1.setdefault("hobby", "sing")
print(dict1)

# 删除2 del前,pop(key)
del dict1["age"]
dict1.pop("hobby")
print(dict1)

# 查找len，看pycharm提示
print(dict1.get("name"))
print(len(dict1))

# 修改update
dict1["height"] = 1.81
dict2 = {"weight": 65,
         "money": 20}
dict1.update(dict2)
dict.clear(dict2)
print(dict1)
print(dict2)

# 遍历
for k, v in dict1.items():
    print(f'{k}--{v}')

# 字符串[]索引下标
num_str = "0123456789"
print(num_str[2:6:1])
print(num_str[2:])
print(num_str[:6:1])
print(num_str[::])
print(num_str[::2])
print(num_str[1::2])
print(num_str[2:-1:])
print(num_str[-2::])
# 逆序
print(num_str[::-1])
str_list = list(num_str)  # 把字符串变为列表
str_list.reverse()
print(str_list)
# 用join接口连接列表
result = ('').join(str_list)
print(result)

a = (1, 2, 3)
b = ('a', 'b', 'c')
result = tuple(zip(a, b))
result1 = list(zip(a, b))
print(result)
print(result1)

seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list2 = list(enumerate(seasons))
# 字典生成式
dict1 = {k: v for k, v in list2}
print(dict1)

dict2 = {v: k for k, v in list2}
print(dict2)


def print_info(name, gender=True, age=""):
    gender_txt = "boy"
    if gender == 0:  # if not gender:
        gender_txt = "girl"
    if age:
        print("%s is a %s age %s" % (name, gender_txt, age))
    print("%s is a %s" % (name, gender_txt))


print_info("li hua")
print_info("wang wu", age="21")
print_info("xiao mei", False, age="18")
