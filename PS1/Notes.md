## Notes 
### 1b
- 1b Logistic Regression Newton's method. 
- We need the logistic regression update rule
- here is the update rule for a single iteration, given that theta is a real number
$$
\
\theta^{(t+1)} = \theta^{(t)} - \frac{J'(\theta)}{J''(\theta)}
\
$$

- if theta is a vector of real numbers, like in logistic regression, this generalizes to 
$$
\
\Theta^{(t+1)} = \Theta^{(t)} - H^{-1} \nabla_\Theta J
\
$$
- H is the Hessian in this case, and $\nabla_\Theta J$ is the gradient vector (same size as theta)
- Thinking about the dimensions:
    - goal is to have shape (1,3)
    - $\nabla_\Theta J$ is going to give shape (1, 3)
    - Hessian is going to be matrix of size (n, n), so (3, 3) in this case 
    - therefore we need to get (1,3) from (1,3) * (3,3), which means (3, 3) * (3,1), we need to transpose $\nabla_\Theta J$

- Hessian is going to be:
$$
\
H = \frac{1}{m} \left[ X^T \cdot g(X\theta) \cdot (1 - g(X\theta)) \right] X
\
$$

- Implementation mistake. There is a huge difference between np.dot (@) and * (elementwise) multiplication. Low level, we can figure out which one is which based on the shape of the operands. If they are the same (broadcastable), we  can use elementwise multiplcation (*). Otherwise if the "inner" shape is the same, we can use np.dot or @ for dot product. To be broadcastable, the shape must be the same or one of them must be 1, which is broadcast to match the other dimension.


### 1e Implement GDA
- GDA is a generative learning model. Models we have looked at earlier are Discriminative learning models, in that they try to learn y given x. Generative learning models try to learn x given y, in other words, they try to fit a curve to the data, and using that fitted curve, tries to make predictions. GDA is trying to fit the data to a gaussian. Specifically, it is trying to fit two separate gaussian distributions, one for each class y = 0 and y = 1.

After finding the values of the parameters of GDA $\phi, \Sigma, \mu_1, \mu_0$, we can find the value of the weights, using the formula:
$$
\theta = \Sigma^{-1} (\mu_1 - \mu_0)
$$
- like in log reg, this will be (n x 1) 

The value of the parameters:
$$
\phi = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}
$$
- this is a probability so (1, 1)

$$
\mu_0 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\}}
$$
- This is average for each parameter, so (n x 1)
- numerator, we multiply a mask (m, 1) with (m. n), resulting in (1, n). We don't worry about summation because it's implicitly done with dot product 
- versus in the denominator, we don't do a dot product, and the mask results in (m, 1). So to reconcile with the summation of 1 to m, we need to do np.sum(mask).

$$
\mu_1 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}
$$
- likewise, as well as the vector operations 
$$
\Sigma = \frac{1}{m} \sum_{i=1}^m \left(x^{(i)} - \mu_{y^{(i)}}\right)\left(x^{(i)} - \mu_{y^{(i)}}\right)^T
$$
- (n x n), its a covariance matrix 
- the value of $\mu$ is either $\mu_1$ or $\mu_2$ depending on the value of y. So if y is 0, then $\mu_0$, likewise if y is 1 $\mu_1$. 
- how to implement this in code? 
- we have two mu, for each operation, we want to choose either mu_0 or mu_1, based on the value of y...
- to do this, we can use np.where()
- **np.where(condition, x_if_true, x_if false)** 
    - allows you to construct new arrays by selecting elements based on a condition
    - condition is a boolean array (0, and 1's)
    - x_if_true: value to take when the boolean array is True (1)
    -x_if_false: value to take when the bollean array is False(1)
    
- **np.sum(), more implementation notes**
- np.sum() adds up numbers in an array. It can sum up all the numbers, or just some of them (based on the axis)
    - np.sum(axis=0): sums up column-wise (vertically)
    - np.sum(axis=1): sums up row-wise (horizontally)
    - we use axis, to compute feature wise values. Each feature only cares about its specific column, so when we do np.sum(), we only want to compute it for each feature.

### 2c)
- deal with dataset 3
- contains one example per row, with x1, x2,y, and t
- first deal with ideal case, where we have access to the true t-labels for training. 
- GOAL: write a log reg classifier with x1 and x2 as inputs, and train it with the t labels (true labels), and ignore y

thoughts:
- we can just throw the log reg module at it

### 2d)
    - done, just did the same things but with labels y instead of t 

### 2e
- estimate constant alpha 
- looks like we can just sum it? Since V+ is the set of all x such that y(i) is 1, so its labelled 
- so we want to make sure that the value of y is 1 

- so how do we apply the "labeling" rate to the value?
- Maybe for all values, we have to reduce  by the labeling rate, to get the actual value 
- so what do we know? We know that 

$$
P(y=1|x) = P(t=1 | x) / \alpha
$$
- so do we just take our current prediction and multiply it by alpha? Correction is just added to the problem. 
- Looks like the best way is to just change the predict function 
- Yep, dividing by the labeling rate solved the issue! 2e is almost done 



### 2e part 2:
- make three separate plots for parts 2c-e. For test set.

### 3d: Poisson regression
- We will apply Poisson regression to model the number of visitors per day. Note that applying Poisson regression in particular assumes that the data follows a poisson distribution whose natural parameter is a linear combination of the input features: 

$$
\eta = \theta^Tx
$$

- from 3c, the update rule for possion regression is 

$$
\theta^{t+1} = \theta - \alpha * (y - e^{\theta^Tx})x
$$

How the heck do we do predict? 
- the canonical response function
- The expected value of y given x is given by:

$$
E[y|x] = g^{-1}(x^T\theta)
$$
- where $g^{-1}$ is the canonical response function. This is because the canonical response function relates the natural parameter to its mean (expected value). Meaning $E[y|x] = g^{-1}(\eta)$. And guess what the assumption about glm is? It's $\eta = \theta^Tx$.

- ok so in poisson regression, the canonical response function is 
$$
e^\eta 
$$
- so the relationship is 
$$
E[y|x] = e^\eta 
$$
- and we know the assumption so, plug that in we get the predict function
$$
E[y|x] = e^{x^T\theta}
$$

### 5b Linear weighted regresssion
- well we know that the update rule is 
$$
\theta^{t+1} = \theta - \alpha \nabla J(\theta)
$$
thoughts:
- we need to create a diagonal matrix, W, so we can use numpy.diag(v), where v is a 1d
vector

- also, the important part is that there is no training loop, the loss function allows us to squeeze out theta, directly from the formula 
- take the loss function
$$
\nabla J(\theta) = 2X^TWX\theta - 2X^TWy
$$

- set it equal to zero, because in convex function this is the global minimum

$$
2X^TWX\theta = 2X^TWy
$$

- isolate theta by multiplying the inverse
$$
\theta = (X^TWX)^{-1}X^TWy
$$
- so for each, all i have to do is find the right weights matrix, and directly plug in the value
-  ok so think, weights matrix equation is 
$$
w^{(i)} = \exp\left( -\frac{\| x^{(i)} - x \|_2^2}{2\tau^2} \right).
$$
- so clearly we use an outer loop for all inner points. There should only be one pivot point, since the pivot point is the predict point
- so it's sealed, hmm, our predict (pivot) point going to be (1, n), and our x^i should also be that
- wait hold up, the predict is also a matrix, so we can't just assume that. So outer loop will be pivot points, and inner loop will be for weights. That'll give us a weights vector for every inner loop, I think...


-- cot 
- X is (200, 2)
- we want weights to be (200, 200) 
- there are 200 pivot points 
- for each pivot point, we create a singular w scalar...
- a single pivot point is (1, 2)
- a single xi is (1, 2)

- we get a w FOR EACH pivot point.
- then use that for prediction
- so pivot point is (1, 2), then our resulting w matrix is the size of x.

- so for each pivot point, we create a separate W. W is a weights matrix the size of m,m, where m is the number of examples in the TRAINING set we kept behind. this works because in the equation for theta, the X is the training set NOT the test set, like we've been initially thinking