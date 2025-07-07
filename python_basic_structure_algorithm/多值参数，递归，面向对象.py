# 作者：hsl
# 邮箱：2049279114@qq.com

def fault(num, age):
    print(f'{num},{age}')


# fault(1,2,3)
# fault(num1=2,age=15)
# fault()
# fault(2,num=15)
def demo(num1, *args, **kwargs):
    print(num1)
    print(args)
    print(kwargs)


demo(1)
a = (1, 2, 5, 4, 7, 6)
b = {"name": "小明", "age": 18}
demo(*a, **b)
print('-' * 50)
# demo(a,b)
my_dict = {'num1': 4}
demo(**my_dict)
demo(*my_dict)
demo(my_dict)


def study_return():
    return 1, 2, 3


tuplex = study_return()
# tuplex,y,z = study_return()
print(tuplex)


class dog():
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def bark(self):
        print("汪汪叫")

    def shake(self):
        print("摇尾巴")


xiaotianquan = dog("小黄", color="yellow")
xiaotianquan.shake()
xiaotianquan.bark()


class Gun:
    def __init__(self, model):
        self.model = model
        self.bullet_count = 0

    def add_bullet(self, count):
        self.bullet_count += count

    def shoot(self):
        # 判断是否有子弹
        if self.bullet_count == 0:
            print("没子弹了，冲啊")
            return

        else:
            self.bullet_count -= 1
            print(f"{self.model}可继续打{self.bullet_count}发子弹")


ak = Gun("ak47")
ak.add_bullet(66)
ak.shoot()


class Soldier:
    def __init__(self, name):
        self.name = name
        self.gun = None

    def fire(self):
        if self.gun is None:
            print(f"冲啊，{self.name}")
            return

        self.gun.add_bullet(20)
        print("向敌人开火")
        self.gun.shoot()


if __name__ == "__main__":
    xusanduo = Soldier("许三多")
    ak47 = Gun("ak47")  # 指Gun类对象
    # xusanduo.fire()

    xusanduo.gun = ak47
    xusanduo.fire()
    print(xusanduo.gun)


class HouseItem:
    def __init__(self, name, area):
        self.name = name
        self.area = area

    def __str__(self):
        return "[%s] 占地 %.2f" % (self.name, self.area)


class House:
    def __init__(self, type, area):
        self.type = type
        self.area = area
        self.free_area = area
        self.item_list = []

    def __str__(self):
        return f"户型：{self.type} 总面积：{self.area} 剩余{self.free_area}： 家具链表：{self.item_list}"

    def add__item(self, item):
        if item.area > self.free_area:
            print("%s 的面积太大，不能添加到房子中" % item.name)
            return
        self.free_area -= item.area
        self.item_list.append(item.name)  # item是地址
        print("要添加 %s" % item)


if __name__ == "__main2__":
    bed = HouseItem("席梦思", 15)
    chest = HouseItem("衣柜", 6)
    table = HouseItem("餐桌", 6)
    print(bed)
    print(chest)
    print(table)

    my_home = House("一室一厅", 25)
    my_home.add__item(bed)
    my_home.add__item(chest)
    my_home.add__item(table)
    print(my_home)


class Women:
    def __init__(self, age):
        self.__age = age

    def __secret(self):
        print(self.__age)

    def friend(self):
        self.__secret()


xiaohong = Women(18)
# 私有属性，外部不能直接访问到
# print(xiaohong.__age)
xiaohong.friend()  # 先返回friend，再返回__secret


class Animal:
    def __init__(self, kind):
        self.kind = kind

    def eat(self):
        print("吃---")

    def drink(self):
        print("喝---")

    def run(self):
        print("跑---")

    def sleep(self):
        print("睡---")


class Dog(Animal):

    def __init__(self, kind, size):
        super().__init__(kind)
        self.size = size

    def shake(self):
        print("摇尾巴")


class Cat(Animal):
    def __init__(self, kind, age):
        super().__init__(kind)
        self.age = age


class XiaoTianQuan(Dog):
    def __init__(self, kind, size, age):
        super().__init__(kind, size)  # king,size形参需要填
        self.age = age

    def fly(self):
        print("飞---")


wangcai = Cat("波斯", "2")

xiaotianquan = XiaoTianQuan("神犬", "big", "1000")
xiaotianquan.shake()


# 子类不能使用父类的私有属性
# 子类不能使用父类的私有属性
# 子类不能使用父类的私有属性

class MusicPlayer:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)  #
        return cls.instance

    def __init__(self):
        print("music viewer")


player1 = MusicPlayer()
player2 = MusicPlayer()
print(player1)
print(player2)