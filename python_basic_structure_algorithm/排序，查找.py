# 作者：hsl
# 2025年06月05日11时37分18秒
# 邮箱：2049279114@qq.com

class Sort:
    def __init__(self):
        self.arr = [3, 87, 2, 93, 78, 56, 61, 38, 12, 40]
        self.arr_len = len(self.arr)

    def partition(self, left, right):
        arr = self.arr
        # 交换i,k k始终指向比分割值arr[right]小的数要放置的位置的下标
        k = left
        for i in range(left, right):
            if arr[i] < arr[right]:
                arr[i], arr[k] = arr[k], arr[i]
                k += 1
        arr[k], arr[right] = arr[right], arr[k]
        return k  # 返回分割点arr[right]的位置参数k

    def partition2(self, left, right):
        arr = self.arr
        # 返回位置参数k
        k = left
        while left < right:
            if left < right and arr[left] < arr[k]:
                left += 1
            elif left < right and arr[right] > arr[k]:
                right -= 1
            else:
                arr[left], arr[right] = arr[right], arr[left]
        return right  # 返回分割点arr[right]的位置

    def quick(self, left, right):
        arr = self.arr
        if left < right:
            pos = self.partition(left, right)
            self.quick(left, pos - 1)
            self.quick(pos + 1, right)

    def adjust_max_heap(self,new_pos,arr_len):
        arr=self.arr
        dad=new_pos
        son=dad*2+1
        while son < arr_len:  # 确保son小于列表长度
            if son + 1 < arr_len and arr[son + 1] > arr[son]:  # 比较左孩子和右孩子，谁大，谁大谁跟父亲比
            # if arr[son + 1] > arr[son] and son + 1 < arr_len:
            # son + 1 < len(arr) 确保 son + 1 在列表索引范围内。
            # 只有当这个条件为真时，才会检查 arr[son + 1] > arr[son]。这样可以避免 IndexError 错误的发生。
                son=son+1
            if arr[dad]<arr[son]:
                arr[dad],arr[son]=arr[son],arr[dad]
                #更新dad和son
                dad=son
                son=dad*2+1
            else:
                break
    def heap(self):
        arr=self.arr
        for dad_pos in range(self.arr_len // 2 - 1, -1, -1):#
            self.adjust_max_heap(dad_pos, self.arr_len)
        #交换堆顶和最后一个元素
        for i in range(self.arr_len-1,0,-1):
            arr[i],arr[0]=arr[0],arr[i]
            self.adjust_max_heap(0, i)


    def half_search(self,target):
        arr=self.arr
        low=0
        high=self.arr_len-1
        while low<=high:
            mid=(low+high)//2
            if arr[mid]>target:
                high=mid-1
            elif arr[mid]<target:
                low=mid+1
            else:
                return mid
        return -1


MAXKEY = 100
tr = [95, 17, 3, 31, 86, 75, 56, 19, 38, 26, 94, 54, 53, 72, 59, 61, 74, 58, 78, 60, 64, 43, 52, 90, 84, 19, 92, 2, 71, 12, 67, 10, 53, 85, 98, 24, 11, 41, 44, 55]
str_list =[str(num) for num in tr]


def hash_search( str_hash):
    h = 0
    g = 0
    for i in str_hash:
        h = (h << 4) + ord(i)
        g = h & 0xf0000000
        if g:
            h ^= g >> 24
        h &= ~g
    return h % MAXKEY

# str_list=["xiongda","lele","hanmeimei","wangdao","fenghua"]
if __name__ == '__main__':
    my_sort = Sort()
    # my_sort.quick(0, my_sort.arr_len - 1)
    my_sort.heap()
    print(my_sort.arr)
    print(my_sort.half_search(78))
    print(my_sort.half_search(12))

    hash_table=[None]*MAXKEY  #初始化一个哈希表
    for i in str_list:
        if hash_table[hash_search(i)] is None:
            hash_table[hash_search(i)]=[i] #第一次放入
        else:
            hash_table[hash_search(i)].append(i) #哈希冲突后拉链法解决
    print(hash_table)