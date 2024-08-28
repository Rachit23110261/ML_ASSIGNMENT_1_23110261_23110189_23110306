# Runtime Complexity Analysis

## Introduction

This document presents the results of the runtime complexity analysis for the decision tree algorithm. We performed experiments with varying numbers of samples (N) and binary features (M) to measure the time taken for learning the tree and predicting test data.

## Experimental Setup

- **Dataset**: Synthetic dataset with varying N (number of samples) and M (number of binary features).
- **Metrics**: Time taken for learning the tree and time for prediction.
- **Implementation**: Custom decision tree algorithm with `GiniIndex` criterion for splitting.

## Results

### Learning Time and Prediction TIme

- **Plot**: 

  ![Learning Time Plot](.\Figure_1.png)

- **Analysis**: 

  The plot compares the experimental learning times against the theoretical learning times for different values of binary features (M). As shown, the experimental learning times generally follow the trend of the theoretical times. The theoretical learning time is approximated by \(N*M*log_2(N)\), reflecting the complexity of processing each sample across all features. Variations between the experimental and theoretical results are expected due to factors like overhead in implementation and machine performance.

  The prediction time plot displays the experimental results compared with the theoretical prediction times. The theoretical time complexity for prediction is approximated by \(N*log_2(N)\), where \(T\) is the number of test samples. The experimental results generally align with the theoretical expectations, demonstrating how the decision tree's prediction time scales with the number of samples. Differences between experimental and theoretical results are attributable to various practical factors such as data structure and implementation specifics.

### Comparison with Theoretical Complexity

- **Theoretical Complexity**:

  - **Learning Time**: \(N*M*log_2(N)\)
  - **Prediction Time**: \(N*log_2(N)\), where \(T\) is the number of test samples.

- **Experimental Observations**:

  The experimental learning times and prediction times were consistent with the theoretical models. Learning times increase with the number of samples and features, following the expected logarithmic pattern. Prediction times scale with the number of test samples and the logarithm of the number of training samples, as predicted. Overall, the results validate the theoretical time complexity models for the decision tree algorithm, with practical discrepancies being within reasonable bounds.

## Conclusion

- The experimental results align well with the theoretical complexity models for both learning and prediction times of the decision tree algorithm.
- The analysis provides insights into how the decision tree's runtime scales with dataset size and feature dimensions, which is crucial for optimizing performance and scalability.
- Future work could explore additional optimizations and comparisons with other machine learning algorithms to further enhance understanding and efficiency.

