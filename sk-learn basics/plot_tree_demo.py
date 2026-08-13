from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# --- Sample training data (replace with your actual data) ---
# Each row: [weight, color] -> color encoded as a number if it's categorical
X = [
    [150, 0],   # apple
    [170, 0],   # apple
    [130, 1],   # banana
    [140, 1],   # banana
    [300, 2],   # mango
    [280, 2],   # mango
]
y = ["apple", "apple", "banana", "banana", "mango", "mango"]

# --- Train the model ---
clf = DecisionTreeClassifier()
clf.fit(X, y)

# --- Plot the tree ---
plt.figure(figsize=(25, 8))
plot_tree(clf, feature_names=["weight", "color"], class_names=["apple", "banana", "mango"])
plt.show()