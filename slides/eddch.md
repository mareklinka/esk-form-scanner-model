## Turning on the lights
-----

### Constructing a computer vision model from scratch

##### _by Marek Linka (ESK)_

---

### Backstory
-----

----

_Once upon a time, at a customer..._ 

Note:
Who owns a car? Vehicle registration certificate?
Customer has an application, wants to speed up processing of the vehicle certificates
Only two fields are relevant so traditional approaches encounter issues

----

![](images/certificate_example.png)

----

So I thought

_Would it be possible to solve this using machine learning?_

Note: Now you have two problems, right?

---

### Laying down the foundations 
-----

----

Before you start implementing, decide what you want to achieve

----

_Given a photo of a vehicle registration certificate, I want to know the coordinates of a bounding box containing the whole matriculation number and nothing else_

----

![](images/certificate_example.png)

----

Machine learning is a wide field - pick an algorithm that suits the problem

Note: Neural nets? This is a vision task, so let's do CNN, as this is the current state of the art. This requires to have basic overview of what's possible.

----

A lot of smart people is doing ML, try searching for solutions to similar problems

----

Adapt an existing algorithm to suit your specific needs

Note: YOLO, region proposal etc. They don't fix exactly, but they can serve as inspiration

----

Computer vision

↓

Convolutional neural networks

↓

YOLO-inspired architecture

---

### Choosing the tools
-----

----

Picking the right tools can save you a lot of time

Note: TensorFlow, CNTK, Caffe, Keras, python, but also hardware

----

Python + TensorFlow-GPU + Keras is a good starting combination

Note: relatively shallow learning curve, good API
nVidia Docker is a good starting point for Linux

---

### Getting the data
-----

----

Data makes or breaks your model

----

If you have a lot of data, awesome.

If you have little data, you have work to do.

Note: manual generation, finding a dataset, transfer learning, data augmentation. You generator needs to do labels as well.

----

Image data is hard

```
def __find_corners(image):
    a = np.array(image).T
    upper_left = np.argwhere((a[2, :, :] == 255) & (a[1, :, :] == 0) & (a[0, :, :] == 255))
    lower_left = np.argwhere((a[2, :, :] == 0) & (a[1, :, :] == 255) & (a[0, :, :] == 0))
    upper_right = np.argwhere((a[2, :, :] == 255) & (a[1, :, :] == 255) & (a[0, :, :] == 0))
    lower_right = np.argwhere((a[2, :, :] == 255) & (a[1, :, :] == 0) & (a[0, :, :] == 110))

    if ((upper_left.size == 0) | (lower_left.size == 0) | (upper_right.size == 0) | (lower_right.size == 0)):
        return ()

    return (upper_left[0], lower_left[0], upper_right[0], lower_right[0])
```

---

### Writing the code
-----

----

### Convolutional layers

```
model.add(Conv2D(32, (5, 5),strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
```

Note: train_model.py
Consists of a convolution, a batch normalization, activation, and pooling

----

Convolution

![](https://i.stack.imgur.com/YDusp.png)

----

Convolution

![](https://www.mathworks.com/content/mathworks/www/en/solutions/deep-learning/convolutional-neural-network/jcr:content/mainParsys/band_copy_copy_14735_1026954091/mainParsys/columns_1606542234_c/2/image.adapt.full.high.jpg/1528781209398.jpg)

Number of filters is increasing, they will serve as feature extractors

----

Batch normalization

Note: Reduces internal covariate shift - less churn between batches leads to faster learning and better convergence

----

Activation

![](https://qph.ec.quoracdn.net/main-qimg-45e888d244ed083a4f5fbb11535e41ac)

Note: relu, tanh, sigmoid, leaky relu, softmax

----

Pooling

![](https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png)

Note: average, max, global versions

----

### Fully-connected layers

```
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(c.prediction_size))
```

----

Flatten

![](https://www.mathworks.com/content/mathworks/www/en/solutions/deep-learning/convolutional-neural-network/jcr:content/mainParsys/band_copy_copy_14735_1026954091/mainParsys/columns_1606542234_c/2/image.adapt.full.high.jpg/1528781209398.jpg)

----

Dense

![](https://qph.ec.quoracdn.net/main-qimg-d8ed8345503ab7c55adcf0f603173f98)

Note: regressor - 8 neurons without activation for 4 X/Y coordinates

---

### Compilation and training
-----

```
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
```

----

Training losses

```
model.compile(loss='mae', ...)
```

Note: mean squared error, mean average error, binary_crossentropy etc.

----

Optimizers

```
model.compile(..., optimizer='adam', ...)
```

Note: gradient descent, batch gradient descent, RMSprop, Adam

----

Metrics

```
model.compile(..., metrics=['mae'])
```

----

Model fitting

```
model.fit_generator(
  gen.infinite_generator(trainig_data_path), 
  epochs=100, 
  steps_per_epoch=240, 
  validation_data=gen.infinite_generator("data\\validation"),
  validation_steps=30)
```

----

Model fitting

![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/05/history_validation_dataset.png)

Note: generators, epochs, validation
Accuracy should go up, validation loss should go down
There will be fluctuations in both

----

In an ideal world, you are now done

_(Spoiler alert: this is not an ideal world)_

---

### Failure is always an option
-----

Note: underfitting, not learning, overfitting, slow learning etc.

----

Underfitting and failure to learn

![](http://srdas.github.io/DLBook/DL_images/HPO6.png)

Note: solve by increasing network size and parameters

----

Overfitting and failure to generalize

![](http://srdas.github.io/DLBook/DL_images/HPO6.png)

Note: solve by decreasing network size, providing more examples, adding regularization etc.

----

Slow learning

Note: solve by experimenting with parameters, better hardware

---

### Knowledge earned the hard way
-----

----

Don't try to solve the whole problem right off the bat

Note: pick a subset of the problem and design building blocks for solving that, then increase your scope

----

Make sure your data generators are correct

Note: example with coordinate 0 will throw the algo off

----

Visualizations are better than numbers

Note: faster and better understanding of learning trends

----

Iteration is the name of the game

Note: the faster you can iterate ideas, the faster you converge to a solution

----

Start your network small, enlarge it just enough to overfit, resolve overfitting

You now have the smallest (and fastest) network for solving your problem

Note: 650 MB network vs 1.2 MB network with the same performance

----

Explore your tools

Note: Keras trickery - checkpointing and live validation

---

### Solving the original problem
-----

----

Attempt 1:

* 52 milion parameters
* 1600 examples
* 100 training epochs
* Training time: 5 hours
* Resulting model size: 600 MB

----

Attempt 2:

* Parameter count: 52m vs 0.1m
* Training examples: 1600 vs 2400
* Epoch time: 100s vs 20s
* Model size: 600MB vs 1.2MB
* Validation loss: 4 (0.5%) vs 2 (0.5%)
* Prediction time: 15ms (650ms on CPU) vs 7ms (20ms on CPU)

----

Examples

---

### Q&A
-----

---

### Resources

* [Machine Learning @ Coursera](https://www.coursera.org/learn/machine-learning)
* [Deep Learning @ Coursera](https://www.coursera.org/specializations/deep-learning)
* [Deep Learning with Keras](https://books.google.sk/books/about/Deep_Learning_with_Keras.html?id=20EwDwAAQBAJ&source=kp_cover&redir_esc=y)
* The Interwebs/GitHub/StackOverflow etc.
* The greatest of learning tools, trial and error

---

### Thank you
-----

![](https://i.stack.imgur.com/VC1TX.png)