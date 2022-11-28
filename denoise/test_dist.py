import random
# a = {'aaa':1, 'bbb':2, 'ccc': 3}
# print(len(a))

a = [random.randint(0,3) for i in range(10)]
b = [1,2,3,4,5,6,7,8]
b.extend(b)
print(b)
# print(a)