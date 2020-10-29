from functools import reduce
from partB import *
import random
from re import *
if __name__ == "__main__":
    print("hello world")


def foo(flag):
    """
    This is a fucking function for HW0
    :param flag: boolean bullshit
    :return: int
    """
    if flag:
        print(1)
    else:
        print(0)


foo(False)

foo(True)


def pouw(x,y):
    print(x ** y)


pouw(2,3)


a=5
b=10


(a,b) = (b,a)

print(a)
print(b)

shani = []
print(shani)
shani.append(1)
shani.append(2)

print(shani)

shani.reverse()

print(shani)

for x in range(2,24):
    print(x)

num_list = [x**2 for x in range(1,12) if x %2 == 0]
print(num_list)

num_list = list(range(1,12))

print(num_list)

lst = num_list[3:10:2]
lst.reverse()
print(lst)


def sum(lst):
    return reduce(lambda x,y: x*y, lst)


print(sum([1,3,3]))

with open('./task9.txt','w') as f:
    f.write('I know how to write')

def magicNumLast(x):
    lst = list(range(1,x))
    p = filter(lambda x: x%2 == 0, lst)
    for num in p:
        print(num)

magicNumLast(6)


def magicNum(x):
    return x == reduce((lambda m, n: m + n), list(filter(lambda y: x % y == 0, list(range(1, x)))))

print(magicNum(6))

p = [x for x in range(1,101)]

print(p)

m = [x for x in range(1,6)]
m_res = list(map(lambda x: x**2, m))
print(m_res)

"""
6 and 28 are magic numbers as you can see
"""
filtered = filter(magicNum, range(2,100))
for a in filtered:
    print(a)

"""
and also that way
"""
print([x for x in range(2,100) if magicNum(x)])


"You can see that I have created a class for point"
y = Point(3, 5)
y.show()

print(y)

w = Point()
w.show()


def randomNum(x):
    random_num = random.randint(1, 1000)
    if random_num > x:
        return 0
    else:
        return random_num


res = randomNum(300)
print(res)


def tasksix(lst, num):
    new_lst = []
    copy_lst = lst
    if num == 0:
        return []
    else:
        for i in range(0,num):
            chosen = random.choice(copy_lst)
            copy_lst.remove(chosen)
            new_lst.append(chosen)
    return new_lst


print(tasksix(list(range(1,11)), 4))


def sumboth(s):
    sub = str(s).split(".")
    first_num = sub[0]
    sec_num = sub[1]
    if first_num.isalpha():
        raise ValueError("Dont insert letters you son of a bitch!")
    return int(first_num) + int(sec_num)


print(sumboth("-10.11"))

