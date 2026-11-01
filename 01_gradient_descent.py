# Gradient descent is just a minimization algorithm. I am trying to get the minimum value of the function f(x) = x^2. I start with x = 4. at x = 4, x^2 = 16. But that doesn't matter I guess. I am trying to find the minima, ie value of x at which x^2 is at its minimum.

x = 4
alpha = 0.4
derivative = lambda x: 2*x

for i in range(200):
    old_x = x  # Just saving the value of x to be printed later, literally no other use of this
    x = x - alpha*derivative(x)
    # I could've used the firangi term "grad" instead of derivatinve, but ehh.
    # like x = x - alphase*grad(x)
    print("Old x : ", old_x, "New x :", x, " | Old f(x) : ", old_x**2, "New f(x) : ", x**2)

# I see the answer is tending towards 0 but never going to exact zero.

# round(x, 4) - can be used to round the values.
