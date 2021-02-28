### `src.prediction`

---

- `network.py`
    - `class Network()`
        - `Meta`
            - Mathematical Formulation
                - the public feed-forward neural network can be thought of as the composition of number of functions $f(x) = f_L(...f_2(f_1(x;w_1);w_2)...),w_L)$.
                - Each function $f_l$ takes as input a datum **$x_l$** and a parameter vector $w_l$ and produces as output a datum $x_l+1$. The parameters, denoted as  $w = \{{w_1, ..., w_L}\}$, are learned from data in order to solve a target problem, in this case, classifying labels in an input image that denotes the input frame of data streamed from sensory input of an autonomous system for perception and for multi-label pixelwise classification in order to gauge motion-planning algorithms.
                - The data $x_1, ..., x_n$ are images in general maps from a lattice to one or more real numbers. Given that the data itself is processed as an input matrix $M$ and the data is processed such that we evaluate it in terms of the subsets of the original image, in order to partition the input such that given local information and clustering can help to classify each pixel with respect to its neighboring pixels. This is done in order to make data processing efficient so that important components of the image that indicate objects can be detected and informed to a different vector stream (e.g. motion planning), for an autonomous system.
                - Given each input image, say $x_i$, will be a $M x N x K$  real array of $M x N$ pixels and $K$ channels per pixel.
                - The input image will undergo a series of transformations.
                - The rectifier activation function, denoted as $y_{ijk} = max\{{0, x_{ijk}}\}$.
                - In terms of region / search space, we would use the 3x3 receptive field, and apply the linear operator and filter to generate the set of feature maps which will then be iterated over with the kernel in the forwardpass to apply max pooling (removing greatest pixel in each 4-pixel subsubmatrix)
                - The weight decay rate was $w$, the momentum was set to $0.9$, and the weight initialization was validated with the precedence property $p_n$ in order to maintain gradient stability during backpropagation, checking for null neuron activations and other inconsistencies of the network during its training process.
                - We initialized the bias, as well as randomizing the weights of the network to then converge towards a local minima with backpropagation and gradient descent. The learning rate was constant throughout the training process.
                - Given the input image matrix is greater than the fixed input `image_size` is `224x224`, other parts of the image will be sent as batches through automatic partitioning.
                - We applied dropout regularization for the trainable fully-connected layers, and we applied batch normalization, data augmentation (random transformations with horizontal flipping, possibly to increase accuracy), in order to further apply nominal neural network optimization techniques.
                - Another important CNN building block is channel-wise normalisation. This operator normalises the vector of feature channels at each spatial location in the input map x.
                - We evaluate the state of the vector norm of each computation during the feedforward pass, and verifying the safety properties given the perturbed input and verifying the input-output state-transition relation.
                - Given that the softmax operator is denoted as $y_{ijk'} = \frac{e^{x_{ijk'}}}{\sum_{k} e^{x_{ijk}}}$, and the log-loss is denoted as $y_{ij} = - \log x_{ij c_{ij}}$, where $c_{ij}$ is the index of the ground-truth class at spatial location $(i,j)$.
                - 
            - Implementation
        - `Implementation`
            - Train the network, initialize the weights (check with precedence property), create the `image_mask`, `image_size`, given object params (epochs=1000, kernel_size=3, stride=1, momentum=0.9, weight_decay=0.0003, optimizer=Adam(learning_rate=0.003)), preprocess the data in `'../../../data/train'` by iterating over each image and applying transformations to augment data for `ImageDataGenerator` and set `batch_size=64`, given `input_image_size=(h=1024, w=2048, channels=3)`,

                ```python
                # setup class to handle augmentation of images and processing workloads 
                train_generator = ImageDataGenerator(rescale=1./255) # down-scale each rgb value to optimize training
                mask_generator = ImageDataGenerator() # setup img_masks given each input_image
                train_iterator = train_generator.flow_from_directory('../../../data/train', class_mode='categorical', batch_size=64, shuffle=False)
                # definitely do object var type checks as well as module type checks 
                # fit model to given image set in generator
                # assume that model is using functional api, dynamic and can handle non-linear operators
                	model.add(Input(shape=(16, ))
                	model.add(Conv2d, (128,128,3)) # define size of each layer given standard DCNN for semantic seg 
                	model.add(layers.MaxPooling2d((3,3), strides=(2,2), padding='same'))
                	model.add(Conv2d, (128, 128, 3))
                	model.add(layers.Dense(64, activation=activations.relu))
                	model.add(Conv2d, (128, 128, 3))
                	model.add(optimizer=Adam(learning_rate=0.003))
                	model.add(layers.UpSampling2D(size=(2,2)) # copy each row vector in current_image iterating in ImageDataGenerator 
                	model.add(layers.Dense(32, activation='softmax') 

                # train network
                model.fit_generator(train_iterator, steps_per_epoch=78.125)
                # evaluate network
                model.evaluate_generator(test_iterator, steps=78.125)

                # encrypt network
                model.load_weights(pretrained_weights)
                # create clone model to encrypt
                tfe_model = tfe.keras.models.clone_model(model) # dual network?

                ```

                - The psuedo-code above is to illustrate the public network, its preprocessing method (most likely need to resize/rescale/crop and convert tf.Tensor to PIL format for ImageDataGenerator())
        - `Network.build_compile_model()`
        - `Network.evaluate()`
- `metrics.py`

### `src.crypto`

---

- `mpc.py`
- `mpc_network.py`

### `src.verification`

---

- `specification.py`
- `main.py`
- `stl.py`

### `src.adversarial`

---

- `adversarial.py`

### `src.deploy`

---

- `main.py`
