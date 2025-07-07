# 作者：hsl
# 邮箱：2049279114@qq.com

class MusicPlayer:
    instance = None
#实例化对象（为对象申请空间，但没有初始化），把对象返回给cls。instance来存储
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)  #
        return cls.instance

    def __init__(self):
        print("music viewer")


player1 = MusicPlayer()
player2 = MusicPlayer()
#地址相同
print(player1)
print(player2)