# 作者：hsl
# 2025年06月04日15时57分22秒
# 邮箱：2049279114@qq.com

# 1.使用Python的队列deque
from collections import deque

import random

# 通常用链表或者列表实现
queue = deque(["Eric", "John", "Michael"])
# 增删查改
queue.append("xionger")
queue.extend(["luke", "you"])
queue.insert(1, "aily")
print(queue)

queue.pop()
queue.popleft()
del queue[2]
queue.remove("aily")
print(queue)

print(queue[1])
queue[0] = "xiongda"
print(queue)


# 2.实现有5个元素的循环队列
class CircleQueue():
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.arr = deque(maxlen=maxsize)  # [0]*maxsize  deque(maxlen=maxsize)
        self.rear = 0
        self.front = 0
        self.num = 0

    def enqueue(self, ele):
        '''
        入队
        :param ele:
        :return:
        '''
        if (self.rear + 1) % self.maxsize == self.front:
            print("队满")
            return False

        if self.num <= self.maxsize - 1:
            self.arr.insert(self.rear, ele)
            self.num += 1

        if self.num > self.maxsize - 1:
            self.arr[self.rear] = ele
            self.num += 1
        self.rear = (self.rear + 1) % self.maxsize
        return True

    def dequeue(self):
        if self.rear == self.front:
            print("队空")
            return False
        element = self.arr[self.front]
        self.front = (self.front + 1) % self.maxsize
        return element


if __name__ == '__main__':
    c = CircleQueue(5)
    c.enqueue(1)
    c.enqueue(2)
    c.enqueue(3)
    c.enqueue(4)
    c.enqueue(5)
    print(c.arr)
    print(c.dequeue())
    print(c.dequeue())
    print(c.arr)
    c.enqueue(5)
    c.enqueue(6)
    c.enqueue(7)
    print(c.arr)


# 3.完成二叉树的层次建树，并实现前序遍历，中序遍历，后序遍历，层序遍历
class Node:
    def __init__(self, ele=-1, left=None, right=None):
        self.ele = ele
        self.left = left
        self.right = right


class Tree:
    def __init__(self):
        self.root = None
        self.queue = deque()  # 初始化一个空的辅助队列

    # 用辅助队列建树
    def insert(self, ele):
        new_node = Node(ele)
        self.queue.append(new_node)
        # 先判断根节点是否为空
        if self.root is None:
            self.root = new_node
        else:
            if self.queue[0].left is None:
                self.queue[0].left = new_node
            else:
                self.queue[0].right = new_node
                self.queue.popleft()  # 父结点满了，队列出队，列表popleft()

    def pre_order(self, node: Node):
        """
        前序遍历就是深度优先遍历
        """
        if node:  # 二叉树非空即队列非空，递归遍历
            print(node.ele, end="")
            self.pre_order(node.left)
            self.pre_order(node.right)

    def in_order(self, Node):
        if Node:
            self.in_order(Node.left)
            print(Node.ele, end="")
            self.in_order(Node.right)

    def post_order(self, node: Node):
        """
        前序遍历就是深度优先遍历
        """
        if node:  # 二叉树非空即队列非空，递归遍历
            self.post_order(node.left)
            self.post_order(node.right)
            print(node.ele, end="")

    def level_order(self):
        # 辅助队列
        queue = []
        queue.append(self.root)
        while queue:
            node1 = queue.pop(0)
            print(node1.ele, end=' ')  # 打印出队元素值
            if node1.left:
                queue.append(node1.left)
            if node1.right:
                queue.append(node1.right)


if __name__ == '__main__':
    tree = Tree()
    for i in range(1, 10):
        tree.insert(i)  # 树的结点插入
    tree.pre_order(tree.root)
    print('\n------------------------')
    tree.in_order(tree.root)
    print('\n------------------------')
    tree.post_order(tree.root)
    print('\n------------------------')
    tree.level_order()
    print()


# 4.完成冒泡排序

class Sort:
    def __init__(self, arr_len):
        self.arr_len = arr_len
        self.arr = []
        self.arr_random()  # 调用下面的arr_random()函数

    def arr_random(self):
        for i in range(self.arr_len):
            self.arr.append(random.randint(0, 100))  # random.randint

    def bubble_sort(self):
        arr = self.arr  # 简便
        for i in range(self.arr_len - 1, 0, -1):  # 最大值为9，最小值1
            for j in range(i):  # 即最大值8，最小值为0
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]


if __name__ == '__main__':
    my_sort = Sort(10)
    print(my_sort.arr)
    my_sort.bubble_sort()
    print(my_sort.arr)
