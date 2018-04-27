## Machine Learning for Document Scanning

##### _by Marek Linka_

---

## The Task
-----

----

## We want to do...

1. A guy comes to you with a vehicle registration certificate
2. You take a picture of the document using your phone/tablet
3. You want to do OCR on the document, but only on a specific field
4. This piece of info is then pre-filled in the application

----

![](images/certificate_example.png)

----

## The Complications

* OCR is a well researched topic, but this is not a classic OCR problem
* Unless you want to feed the whole image into OCR, you want to find the region to scan
* What about if the user takes the picture upside down or with perspective distortion?
* What if the image is low quality/bad lighting?

----

## Traditional solution

1. Preprocess image (desaturate to black and white, correct perspective, resize)
2. Find correct rotation
3. Either
    1. Try finding the target frame or
    2. Run OCR on everything and extract the data you want

----

## Potential issues

1. This solution would be fine for scans, which tend to be relatively consistent
2. Since the problem specifically asks for pictures taken with a phone/tablet, the consistency will be low
3. Therefore the success rate will be relatively low

---

# Machine learning to the rescue!
-----

----

## What is ML?

* In simple terms, we throw a lot of data at an algorithm and we expect it to __learn__ to solve the task for new data
* In reality, machine learning is a lot of
    * Complicated math
    * Complicated algorithms
* The good news is you don't need to understand the gritty details the reap the benefits

----

## Why choose ML for this task?

* Machine learning models are great for
    * Solving complex problems
    * Handling noisy/highly-variable data
    * Extrapolating to new data
    * Adapting to change

----

## Restating the problem

* First step in designing a ML-based solution to a problem is to define the problem properly and fully
* In our case, the problem can be defined as:

_Given a photo of a vehicle registration certificate, I want to know the coordinates of a bounding box containing the whole matriculation number and nothing else_

----

## Choosing the algorithm

* ML is a very wide field, not everything in it will fit our needs
* A good question to ask is _How would a human solve this?_

----

## The human approach to our problem

* Think about the way you think about the problem!
* Hint: you would probably use your eyes, so
    * what are you looking for when you see the picture?
    * how do you locate the area with the matriculation number?

---

# Before you start learning
-----

----

## ML area

* Since we decided humans use eyes to solve our problem, we can look into computer vision algorithms
* Computer vision can be generally split into two categories:
    * Image classification
    * Object detection and localization

----

## Image classification

* These kinds of tasks revolve around looking at a picture and deciding whether it's a cat
    * or it might also be a trash panda
    * or someone's lunch
    * or none of the above

----

## Example

[add]

![](http://www.catster.com/wp-content/uploads/2017/08/A-fluffy-cat-looking-funny-surprised-or-concerned.jpg)
![](https://i.imgur.com/7R9SMwF.jpg)
![](https://images.media-allrecipes.com/userphotos/465x465/3759440.jpg)
![](https://a-static.projektn.sk/2018/04/2018-04-24-MinisterkaSakova-1000x630.jpg)

----

## Image detection and localization

* In these tasks, the algorithm looks at a picture and tried to figure out what's in it
    * Maybe there is a car and a traffic light and another car and a guy
    * We also want to know exactly _where_ these objects are, using a bounding box

----

## Example

![](https://software.intel.com/sites/default/files/managed/a7/55/object-detection-recognition-and-tracking-fig00.jpg)

----

## So we want to find stuff

* Our problem is an image localization problem - we are searching for a specific thing and want to know where it is in the picture
* Object localization is a supervised learning problem
    * We give the algorithm images and correct answers (called _labels_)
    * It will try to learn the relationship between the image and the answer
    * When the answer is a bounding box, it will learn to find bounding boxes

----

## Data

* ML works best when you have _a lot_ of data
* What _a lot_ means depends on your problem, but in general you want to
    * provide enough examples for all of your cases
    * cover edge cases
    * provide enough negative examples (if applicable)

----

## The problem with data

* Unfortunately, the vehicle registration certificates contain protected personal data
* It's not possible to get more than a handful of examples
* If we want a lot of data, we need to create it ourselves
* This process is called _data augmentation_ and is an entirely legit strategy

----

## Data augmentation

* Get a few examples of data
* Figure out what kind of noise will be present in real-life data
    * Rotation, skew, perspective distortion, low resolution etc.
* Write a generator that will take existing examples, applies random noise, and saves the result as a new training example
* The generator must also _label_ the data (save the correct answer)

----

## Data augmentation issues

* It's absolutely crucial the data you generate this way cover your bases
* It's equally crucial your generator is implemented correctly
    * If your labels contain invalid/wrong answers, the learning algorithm will fail in very subtle ways
* Take care when generating the noise - the various types of noise in the generated data should be roughly equivalent to what you expect to see in the wild

---

# Let's learn!
-----

----

## Object detection in practice

* There are several approaches to object detection:
    * Sliding window detector (a classifier repurposed for detection)
    * Region proposal (classifier-based and native)
    * YOLO (native deep net)

----

## Sliding window detector

* Sliding window detector repurposes an object classificatio network into a detector
* It moves a _window_ over an image and feeds the window's content into a classifier
* If the classifier says "yes, this is a thing", you have found a thing
* There are various issues with this approach (such as performance and object sizing)

----

## Example

![](https://www.pyimagesearch.com/wp-content/uploads/2014/10/sliding_window_example.gif)

----

## You Only Look Once

* YOLO is a full-fledged _deep_ network for object localization
* It has several versions, iteratively improving its performance
* It feeds the whole image into the network and get's the objects' locations in the end
* It works well with overlapping objects and scaling and large number of classes
* Default implementation inputs 448x448 images

----

## Example

![](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/21a1654b856cf0c64e60e58258669b374cb05539/2-Figure3-1.png)

----

![](https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png)

----

## The architecture of choice

* It's important to note that none of the above architectures are designed for our use case
* They are made with photos of cats in mind, finding stuff in pictures and videos
* We want to detect one specific region in a well known form
* Therefore I decided to try to design a custom architecture

----

## The form-scanner net

* The network is highly influenced by YOLO/Darknet19 networks (and Andrew Ng)
* The input layer accepts images in 1200x847 resolution
* The network consists of several convolutional layers (feature extractors), followed by a shallow fully connected network (regression calculator)
* The output layer has 8 neurons, each outputting a coordinate X/Y of the bounding box
* In total, the network has ~15m trainable parameters

----

## The form-scanner net

[add image]

----

## Implementing the net

* I chose to go with _TensorFlow_ and _Keras_
* TensorFlow by Google is a complex low-level ML framework with bindings for various languages, capable of running on a GPU
* Keras is a nice high-level, multi-platform ML framework for python, abstracting a lot of the ugliness of TF

----

## The form-scanner net

Let's PYTHON!

----

Did I talk about: Keras, Tensorflow, python generators, regularization, epochs, linear output units, relu activation, loss function, adam optimizer, mse?

----

## Training the net

* Once we have a net, we want to try and train it
* Prepare a training and validation datasets
* Run your model on the training set
* Once training completes, run the model against the validation set and evaluate the results

----

## Training results 1

* Training set size: 350
* Epochs: 40
* Time per epoch: 21s
* Trainable parameters: [add]
* Total time: 14 minutes
* Final training accuracy: 61.0%
* Final training loss: 118

----

## Training results 2

* Training set size: 500
* Epochs: 40
* Time per epoch: 30s
* Trainable parameters: [add]
* Total time: 20 minutes
* Final training accuracy: 77.4%
* Final training loss: 38.6

----

## Interpreting the numbers

* The most important metrics for a training run are accuracy and loss
* Accuracy of 77.4% is decently high
* Training loss of 39 is awesome
* However, these numbers can also indicate _overfitting_

----

## MSE loss
[todo: check equation]
$$MSE(m) \\\\\\
= \frac{\sum\limits_{m}{|(A - A'| + |B - B'| + |C - C'| + |D - D'|)^2}}{||m||} \\\\\\
= 39$$

----

## MSE loss

$$ME(X) \\\\\\
=|A - A'| + |B - B'| + |C - C'| + |D - D'| \\\\\\ 
= \sqrt{39} \\\\\\
= 6.24$$

therefore average difference between our model's prediction and the correct answer is

$$\frac{6.24}{8} = 0.78$$

which is _less than a pixel per coordinate_

----

## Overfitting

* Overfitting happens when the model focuses too well on the training set
* It optimizes heavily on the examples it sees in training and fails to generalize to new data
* There are various methods to counter this
    * Enlarge the training set
    * Add regularization, such as dropout
    * Decrease the network size

----

## Hardware limitations

* Training a large neural network takes time
* It's best to train on a GPU, if at all possible
    * Training the same net on the same examples on the CPU will take 10x - 30x more time
* Training a large neural net also takes a lot of RAM
    * Most datasets will not fit into RAM
    * This requires training sequentially on batches that can fit

---

## So now you have a model
-----

----

## Evaluating the model

* It's important to evaluate the model on data it didn't see in training
* Generate a new batch of test data to feed the trained model and evaluate how well it performed
* Watch for overfitting and underfitting

----

## When learning fails

* The learning process might fail for various reasons
    * Not enough data
    * Unbalanced examples
    * You forgot to normalize your data
    * Overfitting/underfitting
* If this happens, go back to the drawing board and iterate

----

## Why iteration matters

* Machine learning is often more art than science
* You must be able to iterate ideas quickly to evaluate them
    * When inspirations strikes, you don't want to spend time tweaking 573 constants somewhere
* Keras and similar frameworks make this easy by abstracting a lot of the complexities

---

## Did it compile? Ship it!
-----

----

## When it seems to work

* The final step in all this is to integrate the model with your system
* Evaluate the model on production data and see if it still performs well
* Keep an eye on the model and retrain it periodically as more production data becomes available

---

## Q&A
-----

----

## One for you

* What issues do you see in the form scanner implementation?
    * Hint: there are many
