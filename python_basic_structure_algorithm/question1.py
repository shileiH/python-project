# # 作者：hsl
# # 2025年06月04日20时43分24秒
# # 邮箱：2049279114@qq.com


# 2、求两个有序数字列表的公共元素集合
import random
def public_number():
    #列表为可变数据类型
    list1=[]
    list2=[]
    for i in range(10):
        list1.append(random.randint(0,100))
        list2.append(random.randint(0, 100))
    print(f"{list1}\n{list2}")
    print("两者的公共元素为：", set(list1).intersection(list2))#交集
public_number()

# 3、给定一个n个整型元素的列表a，其中有一个元素出现次数超过n / 2，求这个元素

def majorityElement(nums):
    votes = 0
    x=nums[0]
    for num in nums:
        if num == x:
            votes += 1
        else:
            votes -= 1
    if votes > 0:
        print(x)
    else:
        print("没有出现次数超过n/2的元素")
    return x
num=[1,2,3,1,1,4,1,5,1,6,1]
majorityElement(num)

# 4、列表、元组，字典的相同点，不同点有哪些，请罗列
 # 均为容器，列表、元组均可容纳不同的数据类型
# 相比于列表，元组不可原地更改（可以以切片的方式实现间接更改），可以看做一个被冻结的列表。
# 字典以键值对作为元素，存储映射关系。

# 5、将元组 (1,2,3) 和集合 {4,5,6} 合并成一个列表。
tuple1=(1,2,3)
set1={4,5,6}
list1=list(tuple1)+list(set1)
print(list1)

# 6、在列表 [1,2,3,4,5,6] 首尾分别添加整型元素 7 和 0。
list1=[1,2,3,4,5,6]
list1.insert(0,7)
list1.append(0)
print(list1)

# 7、反转列表 [0,1,2,3,4,5,6,7] 。
list1=[0,1,2,3,4,5]
print(sorted(list1,reverse=True))
print(list1[::-1])
print(list1)

# 8、反转列表 [0,1,2,3,4,5,6,7] 后给出中元素 5 的索引号。
list1=[0,1,2,3,4,5,6,7,8]
list1.reverse()
print(list1.index(5))#元素 5 的索引号


# 9、分别统计列表 [True,False,0,1,2] 中 True,False,0,1,2的元素个数，发现了什么？
l1=[True,False,0,1,2]
print(l1.count(True),l1.count(False),l1.count(0),l1.count(1),l1.count(2))

# 10、从列表 [True,1,0,‘x’,None,‘x’,False,2,True] 中删除元素‘x’。
list1=[True,1,0,'x',None,'x',False,2,True]
print(list1.count('x'))
list1.remove('x')
print(list1)

# 11、从列表 [True,1,0,‘x’,None,‘x’,False,2,True] 中删除索引号为4的元素。


# 12、删除列表中索引号为奇数（或偶数）的元素。
# 13、清空列表中的所有元素。
# 14、对列表 [3,0,8,5,7] 分别做升序和降序排列。
# 15、将列表 [3,0,8,5,7] 中大于 5 元素置为1，其余元素置为0。
# 16、遍历列表 [‘x’,‘y’,‘z’]，打印每一个元素及其对应的索引号。
# 17、将列表 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 拆分为奇数组和偶数组两个列表。
# 18、分别根据每一行的首元素和尾元素大小对二维列表 [[6, 5], [3, 7], [2, 8]] 排序。相当于按6,3,2进行排序，除非第一个元素相等，按第二个元素排序。
# 19、从列表 [1,4,7,2,5,8] 索引为3的位置开始，依次插入列表 [‘x’,‘y’,‘z’] 的所有元素。
# 20、快速生成由 [5,50) 区间内的整数组成的列表。
# 21、若 a = [1,2,3]，令 b = a，执行 b[0] = 9， a[0]亦被改变。为何？如何避免？----讲了深COPY和浅COPY再做
# 22、将列表 [‘x’,‘y’,‘z’] 和 [1,2,3] 转成 [(‘x’,1),(‘y’,2),(‘z’,3)] 的形式。
# 23、以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有的键。
# 24、以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有的值。
# 25、以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有键值对组成的元组。
# 26、向字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中追加 ‘David’:19 键值对，更新Cecil的值为17。
# 27、删除字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中的Beth键后，清空该字典。
# 28、判断 David 和 Alice 是否在字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中。
# 29、遍历字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21}，打印键值对。
# 30、若 a = dict()，令 b = a，执行 b.update({‘x’:1})， a亦被改变。为何？如何避免？----讲了深COPY和浅COPY再做（现在别做，容易出错）
# 31、以列表 [‘A’,‘B’,‘C’,‘D’,‘E’,‘F’,‘G’,‘H’] 中的每一个元素为键，默认值都是0，创建一个字典。
# 32、将二维结构 [[‘a’,1],[‘b’,2]] 和 ((‘x’,3),(‘y’,4)) 转成字典。
# 33、将元组 (1,2) 和 (3,4) 合并成一个元组。
# 34、将空间坐标元组 (1,2,3) 的三个元素解包对应到变量 x,y,z。
# 35、返回元组 (‘Alice’,‘Beth’,‘Cecil’) 中 ‘Cecil’ 元素的索引号。
# 36、返回元组 (2,5,3,2,4) 中元素 2 的个数。
# 37、判断 ‘Cecil’ 是否在元组 (‘Alice’,‘Beth’,‘Cecil’) 中。
# 38、返回在元组 (2,5,3,7) 索引号为2的位置插入元素 9 之后的新元组。
