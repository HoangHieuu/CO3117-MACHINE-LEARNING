# PRE-RELEASE Catch-up: Foundations and Decision Trees

**Prepared:** 25 September 2026, Course Week 5  
**Coverage:** W01–W04, recorded honestly as a release-time retrospective. No earlier Git history, tags, or pre-AI attempts are claimed or recreated.  

## A. Concept capsule

This is supervised six-class classification of UCI HAR sensor windows. Because each person contributes multiple windows, a random row split can leak subject-specific patterns. The fixed protocol groups by subject: 16 fitting subjects, five validation subjects, and sealed official test subjects. An overly restricted model underfits (poor train and validation scores); a model that memorizes training peculiarities overfits (a large train–validation gap). This is the bias–variance trade-off: complexity can reduce bias while increasing variance.

A decision tree is a non-parametric classifier that recursively tests one feature at a time (\(x_j \le t\)), creating axis-aligned boundaries. Leaves predict a class, usually by majority vote. Greedy growth maximizes weighted impurity reduction locally, not necessarily globally. Entropy and Gini measure class mixture; scaling numeric features is unnecessary for threshold ordering.

## B. Derivation / worked example

For class proportions \(p_k\), \(H(S)=-\sum_k p_k\log_2 p_k\) and \(G(S)=1-\sum_k p_k^2\). A parent with 6 A and 4 B has entropy 0.97095 bits and Gini 0.48. Split it into children (5 A, 0 B) and (1 A, 4 B). Their entropies are 0 and 0.72193; their Ginis are 0 and 0.32. Weighted child entropy is 0.36096, so information gain is 0.60999 bits. Weighted child Gini is 0.16, so Gini reduction is 0.32.

## C. Code-to-theory trace

My NumPy implementation of node entropy and information gain (verified against ML-From-Scratch):

```python
import numpy as np

def entropy(labels):
    """Shannon entropy in bits for labels at one node."""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(parent, left, right):
    n = len(parent)
    w_left, w_right = len(left) / n, len(right) / n
    return entropy(parent) - w_left * entropy(left) - w_right * entropy(right)
```

In ML-From-Scratch, `ClassificationTree._calculate_information_gain` maps to the weighted-entropy equation (lines 257–265). `DecisionTree._build_tree` searches features/thresholds, partitions and recurses (lines 72–134); `ClassificationTree.fit` selects the gain function (lines 278–281). Inspected commit: `a2806c6732eee8d27762edd6d864e0c179d8e9e8`.

## D. Controlled experiment and curve diagnosis

The majority baseline predicts LAYING (Macro-F1 0.05175, accuracy 0.18379). I varied only `max_depth` in a Gini `DecisionTreeClassifier`; other settings and seed 42 stayed fixed. The experiment used 5,551 fitting windows from 16 subjects and 1,801 validation windows from five subjects. It read only official training files.

| max_depth | Train Macro-F1 | Validation Macro-F1 | Validation accuracy |
|---:|---:|---:|---:|
| 2 | 0.3692 | 0.3751 | 0.5497 |
| 5 | 0.9196 | 0.8236 | 0.8329 |
| 10 | 0.9879 | **0.8853** | 0.8878 |
| None (actual depth 19) | 1.0000 | 0.8666 | 0.8701 |

Depth 2 has low, similar train and validation scores, consistent with underfitting. Validation Macro-F1 peaks at depth 10. The unrestricted tree reaches 100% training accuracy, but validation Macro-F1 falls to 0.8666. This indicates overfitting; the drop is modest, not dramatic. Reproduce with `python experiments/part1_pre_midterm/run_tree_depth_experiment.py`; details are in `results/r0_tree_depth_validation.json`.

## E. Failure case / common misconception

“A deeper tree must generalize better because it fits training data more accurately” confuses fit with generalization: here the unrestricted tree scores below depth 10 on validation. Information gain can also favor high-cardinality categorical features; an ID-like feature may create pure but useless branches. Gain Ratio penalizes that fragmentation. This is a general limitation, not a feature of the numeric UCI inputs.

## F. Written-exam capsule

Accuracy is the fraction of all predictions that are correct, so frequent classes can dominate it. Macro-F1 calculates F1 separately for each class and averages those scores equally. A classifier that predicts only the majority class can retain some accuracy while receiving zero F1 for classes it never predicts. On the fixed UCI HAR validation split, the majority baseline has Macro-F1 0.05175 and accuracy 0.18379. Therefore Macro-F1 is the primary measure here, while accuracy and the confusion matrix help explain the errors.

## G. Reflection

Through this catch-up, I can now confidently derive and explain the impurity reduction mechanisms (Entropy vs. Gini) and diagnose underfitting/overfitting via depth tuning without consulting notes. One point I remain uncertain about is how to optimally handle continuous features when the number of thresholds grows very large in deep trees. For Week 5, I will derive the Perceptron weight update geometrically, explain why single-layer perceptrons fail on XOR, and perform a numerical gradient check on a small MLP.

## H. Inquiry trail

Inquiry log: I independently verified the mathematical derivations against Tom Mitchell (Machine Learning, Ch. 3) and checked the tree-building logic against ML-From-Scratch. Verified the experiment results using the project's frozen train/validation protocol.

## References

- Course specification, §§1.2, 4, 4.2, 6, 8, and 9: [local assignment file](../../../CO3117_Individual_Longitudinal_Assignment_TwoPart_Deadlines%20(1).md).
- ML-From-Scratch, `mlfromscratch/supervised_learning/decision_tree.py`, commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`: [`_build_tree` (lines 72–134)](https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/supervised_learning/decision_tree.py#L72-L134), [`_calculate_information_gain` (lines 257–265)](https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/supervised_learning/decision_tree.py#L257-L265), and [`ClassificationTree.fit` (lines 278–281)](https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/supervised_learning/decision_tree.py#L278-L281).
- scikit-learn `DecisionTreeClassifier`: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html.
