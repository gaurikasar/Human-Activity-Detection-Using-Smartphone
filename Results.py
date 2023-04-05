import matplotlib.pyplot as plt

# Accuracies
accuracies = [95.89, 96.74, 96.26, 91]

# Models
models = ['Logistic Regression', 'Linear SVC', 'Kernel SVM', 'LSTM']

# Create bar chart
plt.bar(models, accuracies, color=['r', 'g', 'b', 'y'])

# Set chart title and axis labels
plt.title('Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')

# Show the chart
plt.show()