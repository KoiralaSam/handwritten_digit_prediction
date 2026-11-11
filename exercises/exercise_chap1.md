# Chapter 1 - Exercises

## Exercise 1 - Sigmoid neurons simulating perceptrons, part I

> Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, c>0.
> Show that the behaviour of the network doesn't change.

---

### Setup

A perceptron with weights $w$, bias $b$, and input $x$ has the following output:

$$
\text{output} =
\begin{cases}
0 & \text{if } w \cdot x + b \le 0 \\
1 & \text{if } w \cdot x + b > 0
\end{cases}
$$

### Proof

When we multiply weights and biases of the perceptron by $c$ we get:

$$
\text{output} =
\begin{cases}
0 & \text{if } cw \cdot x + c \cdot b \le 0 \\
1 & \text{if } cw \cdot x + c \cdot b > 0
\end{cases}
$$

Because,

$$
\begin{aligned}
(cw) \cdot x + cb &= c\,(w \cdot x + b)
\end{aligned}
$$

This shows that multiplying by a cosnstant $c$ doesn't change the sign of the outout value because c > 0

- $w \cdot x + b > 0 \iff c(w \cdot x + b) > 0$
- $w \cdot x + b \le 0 \iff c(w \cdot x + b) \le 0$

So every perceptron gives the **same output** as before. This works for every neuron in the network and thus doesn't change the final
output of the network.

## Exercise 2 - Sigmoid neurons simulating perceptrons, part II

> Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to the
> network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been fixed.
> Suppose the weights and biases are such that w⋅x+b≠0 for the input x to any particular perceptron in the network. Now replace
> all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a positive constant c>0. Show that
> in the limit as c→∞ the behaviour of this network of sigmoid neurons is exactly the same as the network of perceptrons.
> How can this fail when w⋅x+b=0 for one of the perceptrons?

---

### Setup

We are given the same network of perceptrons as above problem, $$w \cdot x \neq 0 $$ for any input x.
Now we change the perceptron to a sigmoid neuron and multiply the weights and biases by a constant $$c > 0$$
We NTS that when, $$ \lim c \to \infty $$
The behavior of the neural network of sigmoid remains the same as that of perceptrons.

<figure align="center">
  <img src="image/exercise_chap1/image.png" width="200">
  <figcaption>Figure 1: Sigmoid function</figcaption>
</figure>

<figure align="center">
  <img src="image/exercise_chap1/1794088398190.png" width="200">
  <figcaption>Figure 1: Your caption here</figcaption>
</figure>

Figure 1: Graph of a sigmoid function

When We multiply the function $w \cdot x + b$ with c, we get

$$
\begin{aligned}
cz = (cw) \cdot x + cb &= c\,(w \cdot x + b)
\end{aligned}
$$

The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So the output of a sigmoid neuron with the scaled weights and bias is:

$$
\sigma(z) = \frac{1}{1 + e^{-c\,(w \cdot x + b)}}
$$

### Proof

#### Step 1: A single neuron

Fix the input $x$ and let $z_0 = w \cdot x + b$. By assumption $z_0 \neq 0$, so there are two cases.

**Case 1: $z_0 > 0$.** The perceptron outputs $1$. For the sigmoid neuron, $c z_0 \to +\infty$ as $c \to \infty$, so $e^{-c z_0} \to 0$ and

$$
\lim_{c \to \infty} \sigma(c z_0) = \lim_{c \to \infty} \frac{1}{1 + e^{-c z_0}} = \frac{1}{1 + 0} = 1
$$

**Case 2: $z_0 < 0$.** The perceptron outputs $0$. For the sigmoid neuron, $c z_0 \to -\infty$ as $c \to \infty$, so $e^{-c z_0} \to +\infty$ and

$$
\lim_{c \to \infty} \sigma(c z_0) = \lim_{c \to \infty} \frac{1}{1 + e^{-c z_0}} = 0
$$

In both cases the sigmoid neuron's output converges to the perceptron's output.

#### Step 2: The whole network

We go layer by layer (induction on the layer).

- **First layer:** the inputs are the fixed network input, which is the same for both networks. By Step 1, every first-layer sigmoid neuron's output converges to the matching perceptron's output as $c \to \infty$.
- **Next layer:** suppose the previous layer's sigmoid outputs $a_c$ converge to the perceptron outputs $a$. For a neuron in this layer, $w \cdot a_c + b \to w \cdot a + b \neq 0$. So for large enough $c$, $w \cdot a_c + b$ has the same sign as $w \cdot a + b$, and the argument $c\,(w \cdot a_c + b) \to \pm\infty$. By the same two cases as Step 1, the neuron's output converges to the perceptron's output.

Repeating this up to the output layer shows that every neuron, and therefore the whole network, behaves exactly like the perceptron network in the limit $c \to \infty$. $\blacksquare$

### When $w \cdot x + b = 0$

> [!WARNING]
> The argument breaks down if $z_0 = w \cdot x + b = 0$ for some perceptron.

Then $c z_0 = 0$ for every $c$, so

$$
\sigma(c z_0) = \sigma(0) = \frac{1}{1 + e^{0}} = \frac{1}{2}
$$

The output stays at $\tfrac{1}{2}$ no matter how large $c$ is. It never approaches $0$ or $1$, so it can't match the perceptron's output (which is $0$ in our convention, since $w \cdot x + b \le 0$ gives $0$). The errors can also propagate to later layers, so the sigmoid network's behaviour can differ from the perceptron network's.

## Exercise 3 - Bitwise representation of a digit

> There is a way of determining the bitwise representation of a digit by adding an extra layer to the three-layer network above.
> The extra layer converts the output from the previous layer into a binary representation, as illustrated in the figure below.
> Find a set of weights and biases for the new output layer. Assume that the first 3 layers of neurons are such that the correct
> output in the third layer (i.e., the old output layer) has activation at least 0.99, and incorrect outputs have activation
> less than 0.01.

---

### Setup

The old output layer has 10 neurons, one per digit. For an input image of digit $d$, neuron $d$ has activation $a_d \ge 0.99$ and every other neuron has activation $a_j < 0.01$.

The new layer has 4 neurons, one for each bit $b_3 b_2 b_1 b_0$ of the digit in binary ($2^4 = 16 \ge 10$). We need a $4 \times 10$ weight matrix $W$ and a bias vector $b$ with 4 entries, so that the new neuron for each bit outputs close to $1$ when the bit is $1$, and close to $0$ when the bit is $0$.

| Digit | $b_3$ | $b_2$ | $b_1$ | $b_0$ |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 1 |
| 2 | 0 | 0 | 1 | 0 |
| 3 | 0 | 0 | 1 | 1 |
| 4 | 0 | 1 | 0 | 0 |
| 5 | 0 | 1 | 0 | 1 |
| 6 | 0 | 1 | 1 | 0 |
| 7 | 0 | 1 | 1 | 1 |
| 8 | 1 | 0 | 0 | 0 |
| 9 | 1 | 0 | 0 | 1 |

### Idea

Each bit neuron should "listen" only to the old output neurons whose digit has that bit set to $1$:

- Weight $20$ from every digit whose bit is $1$.
- Weight $0$ from every digit whose bit is $0$.
- Bias $-10$, so the neuron stays off unless one of its "listened-to" digits is on.

| Bit | Digits with the bit set to 1 |
|:---:|:----------------------------:|
| $b_3$ | 8, 9 |
| $b_2$ | 4, 5, 6, 7 |
| $b_1$ | 2, 3, 6, 7 |
| $b_0$ | 1, 3, 5, 7, 9 |

### Answer

Columns are the old output neurons for digits $0, 1, \dots, 9$. Rows are the new neurons $b_3, b_2, b_1, b_0$.

$$
W =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 20 & 20 \\
0 & 0 & 0 & 0 & 20 & 20 & 20 & 20 & 0 & 0 \\
0 & 0 & 20 & 20 & 0 & 0 & 20 & 20 & 0 & 0 \\
0 & 20 & 0 & 20 & 0 & 20 & 0 & 20 & 0 & 20
\end{bmatrix}
,\qquad
b =
\begin{bmatrix}
-10 \\ -10 \\ -10 \\ -10
\end{bmatrix}
$$

Each new neuron computes

$$
\text{bit}_i = \sigma\!\left(\sum_{j=0}^{9} W_{ij}\, a_j + b_i\right)
$$

### Verification

Take any bit neuron. Its input is $z = \sum_j W_{ij} a_j - 10$, where all activations satisfy $a_j \ge 0$ and all weights are $\ge 0$.

**Case 1: the bit should be 1.** The correct digit $d$ has $W_{id} = 20$ and $a_d \ge 0.99$. The other terms are non-negative, so

$$
z \ge 20 \cdot 0.99 - 10 = 9.8
\quad\Longrightarrow\quad
\sigma(z) \ge \sigma(9.8) \approx 0.99994
$$

**Case 2: the bit should be 0.** The correct digit has weight $0$, so only wrong neurons contribute. At most $5$ neurons (the case for $b_0$) have weight $20$, and each has $a_j < 0.01$, so

$$
z < 5 \cdot 20 \cdot 0.01 - 10 = -9
\quad\Longrightarrow\quad
\sigma(z) < \sigma(-9) \approx 0.00012
$$

In both cases the output is either at least $0.99$ or below $0.01$, so the 4 new neurons output the correct binary representation of the digit. $\blacksquare$

> [!NOTE]
> The answer is not unique. Any large positive weight (instead of $20$) with a bias of about half that size in the negative direction works, as long as the checks in the verification still pass.

## Exercise 4 - Why gradient descent picks the steepest direction

> Suppose we want to make a move $\Delta v$ so as to decrease $C$ as much as possible. This is equivalent to minimizing
> $\Delta C \approx \nabla C \cdot \Delta v$. We constrain the size of the move so that $\|\Delta v\| = \epsilon$ for some small
> fixed $\epsilon > 0$. Prove that the choice of $\Delta v$ which minimizes $\nabla C \cdot \Delta v$ is
> $\Delta v = -\eta \nabla C$, where $\eta = \epsilon / \|\nabla C\|$ is determined by the size constraint $\|\Delta v\| = \epsilon$.
>
> *Hint: Cauchy-Schwarz inequality.*

---

### Setup

We want to solve

$$
\min_{\Delta v} \; \nabla C \cdot \Delta v
\qquad \text{subject to} \qquad
\|\Delta v\| = \epsilon
$$

We assume $\nabla C \neq 0$. If $\nabla C = 0$ we are at a critical point and every direction gives $\Delta C \approx 0$.

> [!NOTE]
> **Cauchy-Schwarz inequality.** For any two vectors $a$ and $b$,
> $$|a \cdot b| \le \|a\|\,\|b\|$$
> with equality if and only if one vector is a scalar multiple of the other.

### Proof

**Step 1: a lower bound.** Apply Cauchy-Schwarz with $a = \nabla C$ and $b = \Delta v$:

$$
|\nabla C \cdot \Delta v| \le \|\nabla C\|\,\|\Delta v\| = \epsilon\,\|\nabla C\|
$$

Removing the absolute value gives

$$
\nabla C \cdot \Delta v \;\ge\; -\epsilon\,\|\nabla C\|
$$

So no allowed step can make $\Delta C$ smaller than $-\epsilon\|\nabla C\|$.

**Step 2: when is the bound reached?** Equality in Cauchy-Schwarz needs $\Delta v = t\,\nabla C$ for some scalar $t$. Then

$$
\nabla C \cdot \Delta v = t\,\|\nabla C\|^2
$$

This equals the lowest possible value $-\epsilon\|\nabla C\|$ only when $t < 0$ (pointing opposite to the gradient) and $|t|\,\|\nabla C\| = \epsilon$, because $\|\Delta v\| = |t|\,\|\nabla C\|$ must equal $\epsilon$. Hence

$$
t = -\frac{\epsilon}{\|\nabla C\|} = -\eta
\qquad\Longrightarrow\qquad
\Delta v = -\eta\,\nabla C
$$

**Step 3: check.** With this choice,

$$
\|\Delta v\| = \eta\,\|\nabla C\| = \epsilon
\qquad\text{and}\qquad
\nabla C \cdot \Delta v = -\eta\,\|\nabla C\|^2 = -\epsilon\,\|\nabla C\|
$$

so the step has the right size and attains the lower bound from Step 1. No other step of length $\epsilon$ does as well, because equality in Cauchy-Schwarz holds only for $t = -\eta$. $\blacksquare$

**Conclusion.** Among all steps of fixed size $\epsilon$, the one pointing opposite to the gradient decreases $C$ the most (to first order), by $\epsilon\,\|\nabla C\|$.

## Exercise 5 - Gradient descent in one dimension

> I explained gradient descent when $C$ is a function of two variables, and when it's a function of more than two variables.
> What happens when $C$ is a function of just one variable? Can you provide a geometric interpretation of what gradient descent
> is doing in the one-dimensional case?

---

### What happens

With one variable, $C = C(v)$ and the gradient is just the derivative: $\nabla C = C'(v)$. The first-order change is

$$
\Delta C \approx C'(v)\,\Delta v
$$

The update rule becomes

$$
v \;\to\; v' = v - \eta\, C'(v)
$$

Now there are only two directions to move in: left or right. The rule picks the one that goes downhill.

| Slope $C'(v)$ | Update | Move |
|:---:|:---:|:---:|
| $> 0$ (curve rising) | $v$ decreases | left |
| $< 0$ (curve falling) | $v$ increases | right |
| $= 0$ | $v$ unchanged | stop |

With a fixed-size step $|\Delta v| = \epsilon$, Exercise 4 reduces to $\Delta v = -\epsilon\,\operatorname{sign}\!\big(C'(v)\big)$. This gives $\Delta C \approx -\epsilon\,|C'(v)|$, which is never positive.

### Geometric interpretation

Picture the graph of $C(v)$ as a hilly landscape and a ball sitting on it.

- $C'(v)$ is the slope of the tangent line at the current point.
- Gradient descent steps in the direction opposite to that slope, so it always moves down the curve.
- The step is proportional to the slope: steep regions give big steps, flat regions give tiny steps.
- Near a point where $C'(v) = 0$ (a minimum, maximum or flat spot), the steps shrink to zero and the process stops.

### Example: $C(v) = v^2$

Here $C'(v) = 2v$, so

$$
v' = v - 2\eta v = (1 - 2\eta)\,v
$$

- If $0 < \eta < 1$, then $|1 - 2\eta| < 1$ and $v$ shrinks toward $0$, the minimum.
- If $\eta$ is too large (for example $\eta > 1$), then $|1 - 2\eta| > 1$ and each step overshoots the minimum and lands farther away on the other side. The process diverges.

> [!WARNING]
> Gradient descent only guarantees going downhill locally. In one dimension it can stop at a local minimum that is not the global one, and it can also stall at a flat inflection point where $C'(v) = 0$.

## Exercise 6 - Online learning vs. mini-batch SGD

> An extreme version of gradient descent is to use a mini-batch size of just 1. That is, given a training input $x$, we update our
> weights and biases according to the rules
> $w_k \to w_k' = w_k - \eta\,\partial C_x / \partial w_k$ and $b_l \to b_l' = b_l - \eta\,\partial C_x / \partial b_l$.
> Then we choose another training input, and update the weights and biases again. And so on, repeatedly. This procedure is known as
> *online*, *on-line*, or *incremental* learning. In online learning, a neural network learns from just one training input at a
> time (just as human beings do). Name one advantage and one disadvantage of online learning, compared to stochastic gradient
> descent with a mini-batch size of, say, 20.

---

### Setup

Both methods update the parameters using the same kind of rule. The only difference is how many training examples are used to estimate the gradient before each update.

| Method | Examples per update | Gradient used |
|:---|:---:|:---|
| Online learning | $1$ | $\nabla C_x$ |
| Mini-batch SGD | $20$ | $\dfrac{1}{20}\displaystyle\sum_{j=1}^{20} \nabla C_{x_j}$ |

The full gradient is $\nabla C = \dfrac{1}{n}\sum_x \nabla C_x$. Both methods estimate it from a sample, and the sample is much smaller (and noisier) for online learning.

### Advantage: more updates and faster, cheaper learning steps

Online learning updates the weights after **every single example**. In one pass over $n$ training examples it makes $n$ updates, whereas mini-batch SGD with size $20$ makes only $n/20$.

- Each update is cheap, because it only needs the gradient from one example.
- The network starts improving immediately, without waiting to process a whole batch.
- It can learn from data as it arrives (a stream of data) and doesn't need to store the whole training set.
- The noise in the updates can also help the network escape shallow local minima.

### Disadvantage: noisy gradient and less efficient computation

The gradient from a single example is a very noisy estimate of the true gradient $\nabla C$.

- A mini-batch of $20$ averages out part of the noise, so each step points more reliably downhill. A single example can send the weights in a direction that is bad for most other examples.
- Because of this, the cost can bounce around instead of decreasing smoothly, and a small learning rate is often needed for stability.
- It is also slower in practice. Mini-batches use matrix operations over all $20$ examples at once, which is much faster on modern hardware (vectorization, GPUs) than processing the examples one at a time.

### Summary

| | Online ($1$) | Mini-batch ($20$) |
|:---|:---|:---|
| Updates per epoch | $n$ | $n/20$ |
| Gradient estimate | Very noisy | Less noisy |
| Cost per update | Cheapest | Higher, but vectorizes well |
| Behaviour of cost | Bounces around | Smoother |

**Answer.** *Advantage:* online learning makes far more frequent updates, and it can learn immediately from one example at a time. *Disadvantage:* each update uses a very noisy gradient estimate, and it loses the speed benefit of vectorized mini-batch computation. $\blacksquare$

## Exercise 7 - Matrix form of a layer's activations

### Setup

Layer 2 has $n$ neurons and layer 3 has $m$ neurons. Let $w = $ `net.weights[1]`, $b = $ `net.biases[1]`, and let $a$ be the activations of layer 2.

- $a$ is an $n\times 1$ vector with entries $a_k$.
- $w$ is an $m\times n$ matrix. $w_{jk}$ is the weight from the $k$th neuron in layer 2 to the $j$th neuron in layer 3.
- $b$ is an $m\times 1$ vector with entries $b_j$.

We want to write $a' = \sigma(wa + b)$ in component form and compare it with rule (4).

### Proof

**Matrix-vector product.** By the definition of matrix multiplication,

$$(wa)_j = \sum_k w_{jk}\,a_k .$$

This is valid because $w$ has $n$ columns and $a$ has $n$ entries. The row index $j$ is the output index, so $wa$ is an $m$-vector. This is why the $jk$ ordering is convenient.

**Adding the bias.**

$$(wa+b)_j = \sum_k w_{jk}a_k + b_j .$$

**Applying $\sigma$ elementwise.** Equation (22) in component form is

$$a'_j = \sigma\Big(\sum_k w_{jk}a_k + b_j\Big) = \frac{1}{1+\exp\big(-\sum_k w_{jk}a_k - b_j\big)}, \qquad j = 1,\dots,m .$$

**Comparison with rule (4).** Rule (4) for a single sigmoid neuron with inputs $x_k$, weights $w_k$ and bias $b$ is

$$\frac{1}{1+\exp\big(-\sum_k w_k x_k - b\big)} .$$

For the $j$th neuron in layer 3:

- Its inputs are the layer-2 activations, so $x_k = a_k$.
- Its weights are the $j$th row of $w$, so $w_k = w_{jk}$.
- Its bias is $b = b_j$.

Substituting gives

$$\frac{1}{1+\exp\big(-\sum_k w_{jk}a_k - b_j\big)} = a'_j .$$

So each component of $a'=\sigma(wa+b)$ equals the output rule (4) gives for the corresponding neuron. Equation (22) computes all $m$ neurons at once with one matrix product. $\blacksquare$

### In code

```python
a_next = sigmoid(np.dot(w, a) + b)  # w: (m, n), a: (n, 1), b: (m, 1)
```
