### Problem 6
- working on get_words (get a normalized list of words from a message string.)
- split a message into words, normalize them, and return the resulting list. Splitting should split on spaces, for normaliziation you should convert everything to lowercase, and return the list of words 
- get_words done

- working on create_dictionary
- create a dictionary mapping words to integer indices. This function should create a dictionary of word to indices using the provided training messages. 
- should use the provided training messages, use get_words to process each message
- ignore rare words, add words to the dictionary if they occur in at least five messages 
- input: list of strings containing SMS messages
- return: python dict mapping words to integers 

- working on transform_text:
    - takes a list of text messages into a numpy array for further processing
    - should create an np array, containing the number of times each word appears in each message
    - each row in resulting should correspond to each message
    - each column should correspond to a word 
    - we should have counts for each word 
    - so num rows should be number of messages
    - and the columns, should be size of word_dictionary

- how would i construct this?

## Naive Bayes 
- now for the actual naive bayes model 
- classifying emails is a braoder set of problems called text classification

### how to represent email
- we will represent an email via a feature vector (so one text, one email, is one vector)
- the length of the feature vector is equal to the number of words in the dictionary (number of features is equal to number of words in dictionary)
- set xi = 1, if dictionary[xi] exists, otherwise it will be 0. 
- feature vector is vocabulary, and vocab is built using a training set, not the entire dictionary 

### Naive bayes assumption
- generative model, we want to model p(x | y)
- we need to make the naive bayes assumption, which is that the xi's are conditionally independent given y. Resulting algorithmn is Naive Bayes classifier. The reason this is important is because otherwise we get a (2^(len(dictionary))) - 1 dimensional parameter vector 
- assumption worded: If I tell you that y = 1 is spam, and I give you an example with y = 1, then telling you the status of word 2087 should have no effect on your belief of word 3981. Ie, 

$$
p(x_{2087}|y) = p(x_{2087}|y, x_{39831})
$$

- this assumption allows us to multiply the probabilities (probability independence means multiply)
- so, assuming the dictionary has 50,000 words 

$$
p(x_1, ..., x_{50000}|y) = \Pi p(x_i|y) 
(i=1...n), where n is the number of words in the dictionary
$$

- Now, you get the join likelihood:
$$
\mathcal{L}(\phi_y, \phi_{j | y=0}, \phi_{j | y=1}) = \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}).
$$


- maximize with respect to $\phi_y$, $\phi_{i|y=0}$, and $\phi_{i|y=1}$ gives you:

$$
\phi_{j | y=1} = \frac{\sum_{i=1}^{m} \mathbf{1} \{ x_j^{(i)} = 1 \land y^{(i)} = 1 \}}{\sum_{i=1}^{m} \mathbf{1} \{ y^{(i)} = 1 \}}
$$
- this is just the fraction of spam emails (y=1) in which word j does appear 

$$
\phi_{j | y=0} = \frac{\sum_{i=1}^{m} \mathbf{1} \{ x_j^{(i)} = 1 \land y^{(i)} = 0 \}}{\sum_{i=1}^{m} \mathbf{1} \{ y^{(i)} = 0 \}}
$$

$$
\phi_y = \frac{\sum_{i=1}^{m} \mathbf{1} \{ y^{(i)} = 1 \}}{m}
$$

- so basically to fit the naive bayes model, we just need to write this into the fit function

* Multinomial 
$$
P(x_j | y) = \frac{\sum_{i=1}^{m} x_j^{(i)} 1\{y^{(i)} = 1\} + \alpha}{\sum_{j=1}^{V} \sum_{i=1}^{m} x_j^{(i)} 1\{y^{(i)} = 1\} + \alpha V}
$$
 
### Naive bayes predict
- having fit all these parameters, to make a new prediction on a new example with features X, we simply calculate 

$$
p(y = 1 | x) = \frac{p(x | y = 1) p(y = 1)}{p(x)}
$$

$$
= \frac{\left( \prod_{i=1}^{n} p(x_i | y = 1) \right) p(y = 1)}
{\left( \prod_{i=1}^{n} p(x_i | y = 1) \right) p(y = 1) + \left( \prod_{i=1}^{n} p(x_i | y = 0) \right) p(y = 0)},
$$



### Laplace smoothing
- what if a word combination was not in the original dataset?
- basically just add 1 to the numerator for each word percentage $\phi_{j|y=0/1}$, and k to the denominator, where k is the number of vocabs, because each token can take on value of belonging to 'spam' or 'not spam'


### - prediction function 



$$
\log P(y=1 \mid \mathbf{x}) \propto \log \phi_y + \sum_{k=1}^{K} x_k \log \phi_{k|y_1}
$$

$$
\log P(y=0 \mid \mathbf{x}) \propto \log (1-\phi_y) + \sum_{k=1}^{K} x_k \log \phi_{k|y_0}
$$

The prediction rule is:

$$
\hat{y} = \begin{cases}
1 & \text{if } \log P(y=1 \mid \mathbf{x}) > \log P(y=0 \mid \mathbf{x}) \\
0 & \text{otherwise}
\end{cases}
$$

### Misc

**good to know** : best way to compare an np array to a labels array
accuracy = (predictions == labels).mean()
    - this gives us accuracy because predictions == labels gives us a boolean array where its (1) if the elementwise values are equal at the same index, and 0 if they are not
    - mean() calculates the mean (sum / total num)

### C: 
Intuitively, some tokens may be particularly indicative of an SMS being a particular class. We can try to get an informal sense of how indicative token i is for the SPAM class by looking at 

$$
log \frac{p(x_j = i | y = 1)}{p(x_j = i | y = 0)} = log( \frac{P(tokeni | email is SPAM)}{P(token i | email is NOTSPAM)})
$$

- thoughts:
ok so we have a formula that can give us the indicativeness. We are given model and dictionary...
- so it looks like we should iterate through every word of the dictionary, and calculate the values, keeping track of the top 
- an idea? Maybe for each, token at index i, we have another list at index i which indicates its indicativeness. 
- for each word we can print it out. Now the question is, how can i figure out the percentages? 
- naive bayes predict gives you p(y=1 |x), so not the other way around 
- isn't this just conditional probability?
- perhaps we can use the previously calculated priors... 
- yes we can just use the priors 
 
- for each word, we can just take the prior values and multiply by the number we divided by
- so for the case of y=1, multiply by the number of y=1 in the training class 
- and for the case of y=0, multiple by the number of y=0 in the training class 
- do we need to multiply the number of y=1 and y=0? No, because all the probabilities are being divided by the same value anyway, so it doesn't make a difference. 

### D: 
Support vector machines are an alternative machine learning model that we discussed in class. We have provided an SVM implementation within src/svm.py. One important part of training an SVM parameterized by an RBF kernel is choosing an appropriate kernel radius. 

Complete the compute_best_svm_radius by writing code to compute the best SVM radius which maximizes accuracy on the validation dataset.

The provided code will use your compute_best_svm_radius to compute and then write the best radius into output/p-6_optimal_radius 

