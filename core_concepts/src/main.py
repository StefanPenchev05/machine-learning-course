# A model is just a function, which follows 
# h: X -> Y
# For a simple regression task, we choose : h(x) = ax + b

def model(x, a, b):
    return a * x + b

x = 4
a = 2
b = 3

y_pred = model(x, a, b)

# So the model predicts 11 for input 4
print(y_pred)

# After this we need a function that calculates the correctness of the prediction
# For regression, one common loss is the squared loss: L(h, x, y) = (h(x) - y)^2

def squared_loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

y_true = 10
y_pred = 11

loss = squared_loss(y_pred, y_true)

# If prediction is far away, loss becomes bigger.
print(loss) 

# But in ML, we don't evaluate only one example. We evaluate many examples.
# R_s(h) = 1/m * \Sigma{i=1} ^ m L(h, x_i, y_i)

def empirical_risk(xs, xy, a, b):
    total = 0
    m = len(xs)
    
    for x, y in zip(xs, xy):
        y_pred  = model(x, a, b)
        total += squared_loss(y_pred, y)
    
    return total / m

xs = [1, 2, 3, 4]
xy = [3, 5, 7, 9]  # follows y = 2x + 1

print(empirical_risk(xs, xy, a=2, b=1))
