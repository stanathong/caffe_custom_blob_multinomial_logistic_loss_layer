# caffe_custom_blob_multinomial_logistic_loss_layer

Implementation of multinomial logistic loss that accepts a probability blob of size NxCxHxW and a label blog size of Nx1xHxC.
The code is based on softmax_loss_layer.hpp/softmax_loss_layer.cpp from https://github.com/TimoSaemann/caffe-segnet-cudnn5, which was a fork of BVLC/caffe.<br>

This was a trial to implement a new layer in Caffe to be used in conjunction with [Segnet](http://mi.eng.cam.ac.uk/projects/segnet).<br>

The original Caffe's Multinomial Logistic Loss layer expects a label blob to have a size of Nx1x1x1 i.e. one label per one image. I would like to customise the probabilies obtained from the Softmax layer so I need a separate Multinomial Logistic Loss layer, and that the use of __SoftmaxWithLoss__ layer is not possible.<br>

Below is an example use of this new layer after the Softmax layer. As you can see, this is equivalent to using _SoftmaxWithLoss_ but allows a flexibility to modify the softmax results before computing loss.<br>

```
layer {
  name: "softmax"
  type: "Softmax"
  bottom: "conv1_1_D"
  top: "softmax"
  softmax_param {engine: CAFFE}
}

layer {
  name: "loss"
  type: "CustomBlobMultinomialLogisticLoss"
  bottom: "softmax"
  bottom: "label"
  top: "loss"
}
```

_NB:_ This repository only contains a CPU implementation.
