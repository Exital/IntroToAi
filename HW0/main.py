from functools import reduce

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

print(magicNum(12))

p = [x for x in range(1,101)]

print(p)

m = [x for x in range(1,6)]
m_res = list(map(lambda x: x**2, m))
print(m_res)