def predict(x, a, b):
    return a * x + b


def compute_loss(xs, ys, a, b):
    total = 0
    m = len(xs)
    for x, y in zip(xs, ys):
        total += (predict(x, a, b) - y) ** 2
    return total / m



# Since we know how to calculate the loss(error)
# We need to find a the best parameters(a,b) for correct predictions
# That's why we will use gradient descent

# What exactly the gradient descent does?
# It minimizes the loss J by adjusting parameters step by step.
# At each step, we compute the gradient of J to see how each parameter affects the loss.
# Then we update parameters in the opposite direction of the gradient:
# theta := theta - alpha * grad(J)


def gradient_descent(xs, ys, learning_rate=0.1, epochs=10000):
    a = 0.0
    b = 0.0
    m = len(xs)
    
    for epoch in range(epochs):
        grad_a = 0.0
        grad_b = 0.0
        
        for x,y in zip(xs, ys):
            error = (a*x + b) - y
            grad_a += error * x
            grad_b += error
        
        grad_a = (2 / m) * grad_a
        grad_b = (2 / m) * grad_b
        
        a = a - learning_rate * grad_a
        b = b - learning_rate * grad_b
        
        if epoch % 100 == 0:
            loss = compute_loss(xs, ys, a, b)
            print(f"Epoch {epoch}: loss={loss:.6f}, a={a:.4f}, b={b:.4f}")
            
    return a, b


xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [3.2, 4.7, 7.4, 8.6, 10.9, 13.5, 14.1, 17.2]

a, b = gradient_descent(xs, ys, learning_rate=0.005, epochs=10000)
print(f"Final parameters: a={a:.6f}, b={b:.6f}")