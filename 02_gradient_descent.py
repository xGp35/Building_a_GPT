def gradient_descent(function_name,f, df, x0, alpha, steps=200):
    print("-------------------------------------")
    print(f"So you have given A {function_name}")
    x = x0
    for i in range(steps):
        x = x - alpha * df(x)

        if abs(df(x)) < 0.001: # This is a stopping condition. If the slope at the new x is less than 0.001, then take it as done. This means the function is nearly flat here, so we are near a minimum. We check the slope, not the step size (alpha * slope), because a tiny alpha would make the step small even on a steep slope. Not foolproof: a flat region (like the bottom of x^4) has a small slope over a wide range.
            print(f"Slope too low to continue. I think you should be happy now.")
            print(f"Converged at step {i+1}: x = {x}.")
            return x
        
        if abs(x) > 1e6: # This is condition for divergence. It did not converge to any value
            print(f"Be sad, your solution did not coenverge. Do better, choose better hyperparameters.")
            print(f"Diverged at step {i+1}: x = {x}")
            return x
    # Third outcome: neither converged nor diverged, we just ran out of steps (e.g. linear keeps sliding downhill, |x| keeps bouncing)
    print(f"Ran out of steps without converging or diverging. Try more steps or a different alpha.")
    print(f"Ended at x = {x:.6f}, f(x) = {f(x):.6f}")
    return x


def gradient_descent(function_name,f, df, x0, alpha, steps=200):
    print("-------------------------------------")
    print(f"So you have given A {function_name}")
    x = x0
    for i in range(steps):
        x = x - alpha * df(x)

        if abs(df(x)) < 0.001: # This is a stopping condition. If the slope at the new x is less than 0.001, then take it as done. This means the function is nearly flat here, so we are near a minimum. We check the slope, not the step size (alpha * slope), because a tiny alpha would make the step small even on a steep slope. Not foolproof: a flat region (like the bottom of x^4) has a small slope over a wide range.
            print(f"Slope too low to continue. I think you should be happy now.")
            print(f"Converged at step {i+1}: x = {x}.")
            print("The x value above is the approximate local minimum near the starting point")
            print(f"f(x) : {f(x)}. This is the value of the function at that local minimum. It is not guaranteed to be the minimum value of the whole function")
            print("As we will be trying to minimize the error function function E, this will be the lowest error we can reach from where we started, not necessarily the lowest possible error. Also, it would be achieved at this particular x.")
            print("🚨🚨 Note: Here we are saying x, but when doing dJ/d_theta, the variable will be theta. And the function J is a function of theta. There x and y will have compeltely different meaning compared to this. There, x is the input feature. We don't take derivative of gradient wrt input features. we take it wrt theta0, theta1, etc. Here the only reason we are using x as the variable along which we are finding minima is because of familiarity wit functions bein written as f(x) in math. There it will be J(theta0, theta1). And our end conditions would be modelled on theta0 and theta1 and J")
            return x
        
        if abs(x) > 1e6: # This is condition for divergence. It did not converge to any value
            print(f"Be sad, your solution did not coenverge. Do better, choose better hyperparameters.")
            print(f"Diverged at step {i+1}: x = {x}")
            return x
    # Third outcome: neither converged nor diverged, we just ran out of steps (e.g. linear keeps sliding downhill, |x| keeps bouncing)
    print(f"Ran out of steps without converging or diverging. Try more steps or a different alpha.")
    print(f"Ended at x = {x:.6f}, f(x) = {f(x):.6f}")
    return x

gradient_descent("linear function",lambda x: 3*x + 2, lambda x: 3, x0=4, alpha=0.1) # This does not have a minima at all. Gradient descent will do fuckall.
gradient_descent("shifted parabola",lambda x: (x - 3)**2, lambda x: 2*(x - 3), x0=-5, alpha=0.1)
gradient_descent("x^4 with start point 4", lambda x: x**4, lambda x: 4*x**3, x0=4, alpha=0.1)
gradient_descent("x^4 with start point 1",lambda x: x**4, lambda x: 4*x**3, x0=1, alpha=0.1)
gradient_descent("|x|: the slope doesn’t shrink near the bottom",abs, lambda x: 1 if x > 0 else -1, x0=4, alpha=0.3)
f  = lambda x: x**4 - 3*x**2 + x
df = lambda x: 4*x**3 - 6*x + 1
gradient_descent("Two valleys, start at 2",f, df, x0=2,  alpha=0.01)
#Starting at 2, gradient descent finds the right valley and stops there. It can’t tell there’s a deeper valley on the other side. That’s a local minimum, and it’s a real issue in neural network training.
gradient_descent("Two valleys, start at -2",f, df, x0=-2, alpha=0.01)

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

gradient_descent("Two valleys, derivative as a python function",f, lambda x: numerical_derivative(f, x), x0=2, alpha=0.01)