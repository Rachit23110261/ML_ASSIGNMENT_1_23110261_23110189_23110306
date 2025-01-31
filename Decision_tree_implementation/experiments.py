import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values
def generate_data(N, M):
    X = np.random.randint(0, 2, size=(N, M))  # Binary features
    y = np.random.randint(0, 2, size=N)  # Binary target
    return X, y

N_values = np.arange(1, 20)
M_values = np.arange(1, 20)
learning_times = []
prediction_times = []

for N in N_values:
    for M in M_values:
        X, y = generate_data(N, M)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        tree_model = DecisionTree(criterion='gini_index')
        
        start_time = time.time()
        tree_model.fit(X, y)
        end_time = time.time()
        learning_times.append((N, M, end_time - start_time))
        
        X_test, _ = generate_data(1000, M)  
        X_test = pd.DataFrame(X_test)
        start_time = time.time()
        tree_model.predict(X_test)
        end_time = time.time()
        prediction_times.append((N, M, end_time - start_time))

learning_times = np.array(learning_times)
prediction_times = np.array(prediction_times)

def theoretical_learning_time(N, M):
    return N * M * np.log2(N)

def theoretical_prediction_time(N, T=1000):
    return T * np.log2(N)

fig, axs = plt.subplots(2, 1, figsize=(18, 12))

for M in M_values:
    times = learning_times[learning_times[:, 1] == M]
    axs[0].plot(times[:, 0], times[:, 2], label=f'Experimental M={M}', marker='o')
    
    theoretical_times = [theoretical_learning_time(N, M) for N in N_values]
    theoretical_times = np.array(theoretical_times)
    axs[0].plot(N_values, theoretical_times / max(theoretical_times) * max(times[:, 2]), 
                label=f'Theoretical M={M}', linestyle='--')

axs[0].set_xlabel('Number of Samples (N)')
axs[0].set_ylabel('Learning Time (seconds)')
axs[0].set_title('Learning Time: Experimental vs Theoretical')
axs[0].legend()
axs[0].grid(True)

for M in M_values:
    times = prediction_times[prediction_times[:, 1] == M]
    axs[1].plot(times[:, 0], times[:, 2], label=f'Experimental M={M}', marker='o')
    
    theoretical_times = [theoretical_prediction_time(N) for N in N_values]
    theoretical_times = np.array(theoretical_times)
    axs[1].plot(N_values, theoretical_times / max(theoretical_times) * max(times[:, 2]), 
                label=f'Theoretical M={M}', linestyle='--')

axs[1].set_xlabel('Number of Samples (N)')
axs[1].set_ylabel('Prediction Time (seconds)')
axs[1].set_title('Prediction Time: Experimental vs Theoretical')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
