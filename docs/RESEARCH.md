# RESEARCH

## Adversarial Regularization
The goal is to assess what optimization techniques (as regularization) help model convergence at the client-level to start answering the question with what strategy it works best with in order to help the server-side model against norm-bounded perturbation attacks during evaluation.

Theoretically, you can make relationships between the ideas of surface variations as non-uniform perturbations that help the models converge well without overfitting. But you can also argue that forms of "data augmentation" done through the technique formalized by neural structured learning (which generates non-convex norm-bounded sets based on the input sample as neighbors to regularize the model). You can then relate these ideas to the data and the strategy. For an example, if your aggregation strategy to update the server model with the least amount of data (sparsity) and computation (in a production system, this is crucial), then relating both the ideas of the most optimal adversarial regularization technique and what strategy it works best with based on how it configures client-level data can help build more robust federated learning systems in general.

## Corruption Regularization
The first IMAGENET-C corruption is Gaussian noise. This corruption can
appear in low-lighting conditions. Shot noise, also called Poisson noise, is electronic noise caused by
the discrete nature of light itself. Impulse noise is a color analogue of salt-and-pepper noise and can be
caused by bit errors. Defocus blur occurs when an image is out of focus. Frosted Glass Blur appears
with “frosted glass” windows or panels. Motion blur appears when a camera is moving quickly.
Zoom blur occurs when a camera moves toward an object rapidly. Snow is a visually obstructive
form of precipitation. Frost forms when lenses or windows are coated with ice crystals. Fog shrouds
objects and is rendered with the diamond-square algorithm. Brightness varies with daylight intensity.
Contrast can be high or low depending on lighting conditions and the photographed object’s color.
Elastic transformations stretch or contract small image regions. Pixelation occurs when upsampling a
low-resolution image. JPEG is a lossy image compression format that increases image pixelation and
introduces artifacts. Each corruption type is tested with depth due to its five severity levels, and this
broad range of corruptions allows us to test model corruption robustness with breadth.

The first corruption type is Gaussian noise. This corruption can appear
in low-lighting conditions. Shot noise, also called Poisson noise, is electronic noise caused by the
discrete nature of light itself. Impulse noise is a color analogue of salt-and-pepper noise and can be
caused by bit errors. Defocus blur occurs when an image is out of focus. Frosted Glass Blur appears
with “frosted glass” windows or panels. Motion blur appears when a camera is moving quickly. Zoom
blur occurs when a camera moves toward an object rapidly. Snow is a visually obstructive form of
precipitation. Frost forms when lenses or windows are coated with ice crystals. Fog shrouds objects
and is rendered with the diamond-square algorithm. Brightness varies with daylight intensity. Contrast
can be high or low depending on lighting conditions and the photographed object’s color. Elastic
transformations stretch or contract small image regions. Pixelation occurs when upsampling a lowresolution image. JPEG is a lossy image compression format which introduces compression artifacts.


## Research Notes
- math is involved around decision formulations that use formal notation of functions used on client-level, the federated strategy, the nsl-specific math, and the formalization of different perturbations/configs as regularization etc
- fedadagrad adaptability + feature decomposition from NSL / higher dimensionality of features + DCNN with skip connections and nominal regularizations etc --> converge to satisfy robustness specifications and conform to optimal optimization formulation
- structured signals (higher dimensionality of feature representation) + adaptive federated optimization --> more robust model with corrupted and sparse data;
- strategy (when explaining/comparing GaussianNoise and without GaussianNoise): Other implementations apply the perturbation epsilon deeper into the network, but for maintaining dimensionality (and other reasons specified in the paper), the earlier the perturbations applied, the better (Goodfellow, et. al --> adversarial examples paper).
- adversarial neighbors and NSL core protocol relation to fedAdagrad
- how does convexity apply to the optimizer used to most efficiently aggregate the learnings of each client on local data? Surely important considering optimization formulation is interlinked with specifications that depend on measuring variability.

