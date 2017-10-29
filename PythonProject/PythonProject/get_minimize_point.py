import numpy as np

def slope_given_x(x):
    return 2*(x)

current_x = 1
learning_rate = 0.1
while np.absolute(current_x) > 0.0001 :
    previouse_x = current_x
    current_x += -learning_rate*slope_given_x(previouse_x)
    print(previouse_x)

print("the final result is:" , current_x)
