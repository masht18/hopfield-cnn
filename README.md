# hopfield-cnn

Final project for [2022 IBRO-Simons Computational Neuroscience Imbizo](https://imbizo.africa/archive/isicni2022/). An experiment working with associative memory.

CNN with hopfield final layer. The idea was to store low-dimensional representations of images after they were processed by a CNN, and retrieve the closest stored pattern to the test image during inference. In reality, the hopfield layer retrieves all stored patterns that share a representing neuron, resulting in unrecognizable hybrid patterns (intrusion). 

[[/images/hopfield_intrusion.PNG]

A demonstration of classic Hopfield networks in action, specifically the problems caused by their low capacity.
