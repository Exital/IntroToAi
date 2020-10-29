

class MyClass:
    def __init__(self):
        print("this is my class")


class Point:
    def __init__(self, x = 0 , y = 0):
        self.x_pos = x
        self.y_pos = y


    "we sont need the self inside"
    def show(self):
        print(f"my location is {self.x_pos},{self.y_pos}")


