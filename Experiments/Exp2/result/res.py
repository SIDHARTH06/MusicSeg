import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Given data
data = [
    0.7431324411925163, 0.8014184397163121, 0.7559377997425558,
    0.7730496453900709, 0.7614133942356721, 0.7730496453900709,
    0.779370822706242, 0.7659574468085106, 0.779370822706242,
    0.7659574468085106, 0.7853719138137161, 0.7659574468085106,
    0.7853719138137161, 0.7659574468085106, 0.7830323416212026,
    0.75177304964539, 0.7822501138364718, 0.7446808510638298,
    0.7822501138364718, 0.7446808510638298
]

# Initialize precision and recall arrays
precision = []
recall = []

# Separate odd and even values into precision and recall arrays
for i,value in enumerate(data):
    if i % 2 == 0:
        recall.append(value)
    else:
        precision.append(value)


# Plot precision-recall curve
plt.plot(recall, precision)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.savefig('prc_30.png')