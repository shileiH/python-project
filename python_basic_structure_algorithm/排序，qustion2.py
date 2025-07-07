# 作者：hsl
# 2025年06月06日11时32分51秒
# 邮箱：2049279114@qq.com
# sort 函数是 list 列表中的函数，而 sorted 可以对 list 或者 iterator 进行排序

from operator import itemgetter, attrgetter

list1 = [1, 3, 2, 4, 5, 3, 2]
print(sorted(list1))
print(list1)
list1.sort()#Sort the list
print(list1)


dict1 = {1: 'D', 3: 'B', 2: 'B', 4: 'E', 5: 'A'}
print(sorted(dict1))

str_list = "This is a test string from Andrew".split()
print(str_list)
print(sorted(str_list))

def compare_function(str1: str):
    return str1.lower()

print(sorted(str_list,key=compare_function))
# print(sorted(str_list,key=str.lower)) #key=str.lower,将字符串按首字母小写比较


#可以用lamba表达式

# 按shift+alt 竖选
student_tuples = [
    ('john', 'A', 15),
    ('jane', 'B', 12),
    ('dave', 'B', 10),
]
print('-'*100)
print(sorted(student_tuples , key=lambda student:student[2]))
print(sorted(student_tuples , key=lambda x:x[1]))
print(sorted(student_tuples, key=itemgetter(2), reverse=True))
class Student:

    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age

    def __repr__(self): #repr好处可以返回元组类型
        """
        repr好处可以返回元组类型
        :return:
        """
        return repr((self.name, self.grade, self.age))

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]

print('-'*100)
print(sorted(student_objects, key=lambda x:x.age))
print(sorted(student_objects, key=attrgetter('grade')))
print(sorted(student_objects, key=attrgetter('age'),reverse=True))#attrgetter, 按类中的属性排序

print('-'*100)
print(sorted(student_tuples, key=itemgetter(1,2)))
print(sorted(student_objects, key=attrgetter('grade','age')))
print(sorted(student_tuples, key=lambda x:(x[1],x[2]))) #x:(x[1],x[2])),  按元组()/列表[]中的元素排序,
print(sorted(student_objects, key=lambda x:(x.grade,x.age),reverse=True))

#稳定排序
data = [('red', 1), ('blue', 1), ('red', 2), ('blue', 2),('black',1)]
print(sorted(data, key=itemgetter(0)))
print(sorted(data, key=itemgetter(1)))

mydict = { 'Li'   : ['M',7],
           'Zhang': ['E',2],
           'Wang' : ['P',3],
           'Du'   : ['C',2],
           'Ma'   : ['C',9],
           'Zhe'  : ['H',7] }
print(sorted(mydict.items(), key=lambda x:x[1][1])) #value 结构 [n,m] 中的 m
print(sorted(mydict.items(), key=lambda x:x[1][0])) #value 结构 [n,m] 中的 n
print(sorted(mydict.items(), key=lambda x:x[0][-1]))#key键结构 字符串中的最后一个字母
print(sorted(mydict.items(), key=lambda x:x[0][1]))#key键结构 字符串中的第二个字母
print(sorted(mydict.items(), key=lambda x:x[0][0]))#key键结构 字符串中的第一个字母

gameresult = [
    { "name":"Bob", "wins":10, "losses":3, "rating":75.00 },
    { "name":"David", "wins":5, "losses":5, "rating":57.00 },
    { "name":"Carol", "wins":4, "losses":5, "rating":57.00 },
    { "name":"Patty", "wins":9, "losses":3, "rating": 71.48 }]
print(sorted(gameresult, key=lambda x:x['rating']))
print(sorted(gameresult, key=lambda x:(x['rating'],x['wins'])))
print(sorted(gameresult, key=lambda x:(-x['rating'],x['wins'])))

tuples1=[(3,5),(1,2),(2,4),(3,1),(2,2),(1,3)]
print(sorted(tuples1,key=lambda x:(x[0],-x[1])))


#统计词频
file2=open("Beauty and the Beast",'r+',encoding='utf-8')
str1=file2.read()
list1=str1.split()
file2.close()
print(list1)


# # 使用普通字典来计数
count_dict = {}
for string in list1:
    if string in count_dict:
        count_dict[string] += 1
    else:
        count_dict[string] = 1
print(sorted(count_dict.items(), key=lambda x:x[1], reverse=True))
result=sorted(count_dict.items(), key=lambda x:x[1], reverse=True)
print(result[:10])
x=result[:10]

num=[i[0] for i in x]
print(num)
