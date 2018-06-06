## Machine Learning @ Viacar
### An Overview
-----

##### _by Marek Linka_

---

## The Task
-----

----

### We want to do...

1. A guy comes to us with a vehicle registration certificate
2. We take a picture of the document using your phone/tablet
3. We want to do OCR on the document, but only on a specific field
4. This piece of info is then pre-filled in our application

----

![](images/certificate_example.png)

----

### The complications

* OCR is a well researched topic, but this is not a classic OCR problem
* Unless you want to feed the whole image into OCR, you want to find the region to scan first
* What about if the user takes the picture upside down or with perspective distortion?
* What if the image is low quality/bad lighting?

----

### Traditional solution

1. Preprocess image (desaturate to black and white, correct perspective, resize)
2. Find correct rotation
3. Either
    1. Try finding the target frame or
    2. Run OCR on everything and extract the data you want

----

### Potential issues

1. This solution would be fine for scans, which tend to be relatively consistent
2. Since the problem specifically asks for pictures taken with a phone/tablet, the consistency will be low
3. Therefore the success rate will be relatively low

---

## Machine learning to the rescue!
-----

----

### Why choose ML for this task?

* Machine learning models are great for
    * Solving complex problems
    * Handling noisy/highly-variable data
    * Extrapolating to new data
    * Adapting to change

----

### Restating the problem

* First step in designing a ML-based solution to a problem is to define the problem properly and fully
* In our case, the problem can be defined as:

_Given a photo of a vehicle registration certificate, I want to know the coordinates of a bounding box containing the whole matriculation number and nothing else_

---

## Solving the problem
-----

----

### Image detection and localization

* In these tasks, the algorithm looks at a picture and tries to figure out what's in it
    * Maybe there is a car and a traffic light and another car and a guy
    * We also want to know exactly _where_ these objects are, using a bounding box

----

### Example

![](https://software.intel.com/sites/default/files/managed/a7/55/object-detection-recognition-and-tracking-fig00.jpg)

----

### The problem with data

* Unfortunately, it's difficult to get real certificates
* If we want a lot of data, we need to create it ourselves
* This process is called _data augmentation_ and is an entirely legit strategy

----

### Data augmentation

* Get a few examples of data
* Figure out what kind of noise will be present in real-life data
    * Rotation, skew, perspective distortion, low resolution etc.
* Write a generator that will take existing examples, applies random noise, and saves the result as a new training example
* The generator must also _label_ the data (save the correct answer)

---

## Let's learn!
-----

----

### Object detection in practice

* There are several approaches to object detection:
    * Sliding window detector (a classifier repurposed for detection)
    * Region proposal (classifier-based and native)
    * YOLO (native deep net)

----

### The architecture of choice

* It's important to note that none of the above architectures are designed for our use case
* They are made with photos of cats in mind, finding stuff in pictures and videos
* We want to detect one specific region in a well known form
* Therefore I decided to try to design a custom architecture

----

### The form-scanner net

* The network is highly influenced by YOLO/Darknet19 networks (and Andrew Ng)
* The input layer accepts grayscale images in 400x400 resolution
* The network consists of several convolutional layers, followed by a shallow fully connected network
* The output layer has 8 neurons, each outputting a coordinate X/Y of the bounding box
* The network has ~0.1m trainable parameters

----

### Implementing the net

* I chose to go with _TensorFlow_ and _Keras_
* TensorFlow by Google is a complex low-level ML framework with bindings for various languages, capable of running on a GPU
* Keras is a nice high-level, multi-platform ML framework for python, abstracting a lot of the ugliness of TF

----

### Network details

* Parameter count: 0.1m
* Training examples: 2400
* Epoch time: 20s
* Total training time: ~ 30 minutes
* Model size: 1.2MB
* Validation loss: 2 (0.5%)
* Prediction time: 7ms (20ms on CPU)

----

### Optimization and tricks

* Overfitting is an issue with neural networks
* Dealing with overfitting:
    * _Batch normalization_
    * Dropout
    * Regularization
* Interesting Keras features:
    * Computing validation set loss after every epoch
    * Model checkpointing
* These two can help produce a robust model

---

## Examples
-----