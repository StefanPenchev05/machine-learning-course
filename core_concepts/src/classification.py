from collections.abc import Callable

# Simple calssification
def classifier(x: int) -> int:
    if x >= 5:
        return 1
    return 0

# Classification loss

def zero_one_loss(y_pred: int, y_true: int) -> int:
    return 0 if y_pred == y_true else 1

def empirical_classification_risk(xs, ys, clf: Callable[[int], int]):
    total = 0
    for x, y in zip(xs,ys):
        total += zero_one_loss(clf(x), y)
        
    return total / len(xs)

xs = [1, 2, 5, 7, 3, 10]
ys = [0, 0, 1, 1, 0, 1]

correct = 0
for x, y in zip(xs, ys):
    pred = classifier(x)
    if pred == y:
        correct += 1

# Accuracy is fraction of correct classifications
accuracy = correct / len(xs)
print("Accuracy:", accuracy)

# Empirical risk is fraction of wrong classifications
risk = empirical_classification_risk(xs, ys, classifier)
print("Empirical risk:", risk)