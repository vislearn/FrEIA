Tips & Tricks, FAQ
===============================

* Stochastic gradient descent will not work (well) for INNs. Use e.g. Adam instead.
* Gradient clipping can be useful if you are experiencing training instabilities, e.g. use ``torch.nn.utils.clip_grad_norm_``
* Add some slight noise to the inputs (order of 1E-2). This stabilizes training and prevents sparse gradients,
  if there are some quantized or perfectly correlated input dimenions

For coupling blocks in particular:

* Use Xavier initialization for the weights. This prevents unstable training at the start.
* If your network is very deep (>30 coupling blocks), initialize the last layer in the subnetworks to zero.
  This means the INN as a whole is initialized to the identity, and you will not get NaNs at the first iteration.
* Do not forget permutations/orthogonal transforms between coupling blocks.
* Keep the subnetworks shallow (2-3 layers only), but wide (>= 128 neurons/ >= 64 conv. channels)
* Keep in mind that one coupling block contains between 4 and 12 individual convolutions or fully connected layers.
  So you may not have to use as many as you think, else the number of parameters will be huge.
* This being said, as the coupling blocks initialize to roughly the identity transform,
  it is hard to have too many coupling blocks and break the training completely
  (as opposed to a standard feed-forward NN).

For convolutional INNs in particular:

* Perform some kind of reshaping early, so the INN has >3 channels to work with
* Coupling blocks using 1x1 convolutions in the subnets seem important for the quality,
  they should constitute every other, or every third coupling block
