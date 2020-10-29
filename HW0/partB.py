

class MyClass:
    def __init__(self):
        print("this is my class")


class Point:
    def __init__(self, x = 0 , y = 0):
        self.x_pos = x
        self.y_pos = y

    def __str__(self):
        return f"Y is {self.y_pos}, x is {self.x_pos}"



    "we sont need the self inside"
    def show(self):
        print(f"my location is {self.x_pos},{self.y_pos}")