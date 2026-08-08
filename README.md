# SE4050 – Deep Learning – Lab Sheet 2

## Lab 2: Introduction to Backpropagation and Feed-Forward Neural Networks

---

## Exercise 1: Backprop.ipynb

**Objective**
Understand a manual backpropagation implementation and observe how increasing the number of iterations (epochs) affects prediction accuracy.

**Setup**

| | |
|---|---|
| Environment | Anaconda (conda env `se4050`, Python 3.10) |
| Packages | `tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`, `notebook` |

**What was done**

The notebook trains a simple 2-2-2 feed-forward network (2 inputs, 2 hidden units, 2 outputs) using manual forward propagation and backpropagation, based on Matt Mazur's step-by-step backpropagation example. The `no_of_iter` parameter inside `initialize()` was increased from its default value of 100, and the notebook was rerun at three different values to compare results.

**Results**

| Iterations | Final Output | Desired Output | Error |
|:---:|:---:|:---:|:---:|
| 100 | [0.178, 0.8771] | [0.01, 0.99] | 0.0205 |
| 1,000 | [0.0447, 0.9569] | [0.01, 0.99] | 0.00115 |
| 10,000 | [0.0162, 0.9839] | [0.01, 0.99] | 0.0000379 |

<table>
<tr>
<td align="center"><b>100 iterations</b><br><img src="100.png" width="280"></td>
<td align="center"><b>1,000 iterations</b><br><img src="1000.png" width="280"></td>
<td align="center"><b>10,000 iterations</b><br><img src="10000.png" width="280"></td>
</tr>
</table>

**Observation**

Increasing the number of iterations significantly improved the network's prediction accuracy. At 100 iterations, the error was 0.0205 with output [0.178, 0.8771]. Increasing to 1,000 iterations reduced the error to 0.00115, and at 10,000 iterations the error dropped further to 0.0000379, with the output [0.0162, 0.9839] nearly matching the desired output [0.01, 0.99]. This confirms that each backpropagation iteration adjusts the weights slightly in the direction that reduces error, so more iterations allow the network to converge closer to the optimal weights.

---

## Exercise 2: NN_sample.ipynb

Exercise 2: NN_sample.ipynb - Tuning Hidden Layer Size

Results:
Hidden Units | Accuracy
1            | 67.5%
2            | 67.25%
3            | 90.75%
4            | 90.5%
5            | 91.25%
20           | 90.0%
50           | 90.75%

Question 1: What happens when the number of hidden nodes increase?

Accuracy jumps sharply from 1-2 hidden units (around 67%) to 3+ hidden units
(around 90%), then stays roughly flat between 90-91% all the way from 3 up
to 50 hidden units. It does not keep increasing indefinitely as hidden
units are added.

Question 2: Can you explain the pattern of the accuracy when the hidden
nodes increase?

With only 1-2 hidden units, the network does not have enough capacity to
model the decision boundary of this dataset, so it underfits and gets
stuck around 67% accuracy. Once the model has 3 or more hidden units,
there is enough capacity to fit the data well, and accuracy plateaus
around 90-91%. Adding many more units beyond that (20, 50) does not
improve results further, since the extra capacity is not needed for this
problem and can even risk overfitting on a small dataset rather than
improving generalization.

---

## Exercise 3: MLP_with_MNIST_dataset.ipynb

*(to be added)*

---

## Repository Contents

| File | Description |
|---|---|
| `Backprop.ipynb` | Modified notebook for Exercise 1 (10,000 iterations, results visible) |
| `image.png` | Network diagram used by Backprop.ipynb |
| `100.png`, `1000.png`, `10000.png` | Output screenshots at each iteration count |
| `NN_sample.ipynb` | Modified notebook for Exercise 2 |
| `ex2_answers.txt` | Written answers to Exercise 2 questions |
| `MLP_with_MNIST_dataset.ipynb` | Modified notebook for Exercise 3 |
