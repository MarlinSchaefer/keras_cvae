# Keras CVAE

The code contains a base-class called CVAE in CVAE.py. Furthermore,
a few layers are provided to allow to interface easily with
tensorflow_probability distributions.

To create a CVAE three networks need to be specified: the encoder
network E1 (often called r1), the recognition network E2 (often called
q) and the decoder network D (often called r2). These models may be any
Keras model. They only need to abide to the input- and output shapes
required by the distributions. (See comments in the init-method of the
CVAE class).

The output of the CVAE provides parameters to a set of user-specified
distributions. These distributions are specified by the
output_distributions argument. See the documentation of
CVAE.MultiDistribution for details.

The CVAE may be saved using its save method. Loading currently requires
the dedicated load_cvae method.

## Example
The script train_cvae provides a baseline example. It mostly
reimplements the network provided in [this paper](https://arxiv.org/abs/1909.06296).

In the beginning the output distributions are defined. Afterwards the
three distinct networks for the CVAE are defined. The get_handler method
specifies how the data should be loaded. By default it looks for data
in the subdirectory training_data_corr in the current directory. The
files are expected to be HDF-files that contain at least two dataset 
'y_data_noisefree' (whitened pure waveforms that are NOT scaled) and
'x_data' (the true values). Right now the network outputs the
right ascension and declination in the form of a unit vector in 3D. See
[https://github.com/hagabbar/vitamin_b](https://github.com/hagabbar/vitamin_b)
for details on the implementation of the data.

Finally, the model is trained and saved on every epoch. It also shows
that general Callbacks are supported.
