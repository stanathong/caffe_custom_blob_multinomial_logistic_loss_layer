# caffe_custom_blob_multinomial_logistic_loss_layer
Implementation of multinomial logistic loss that accepts a probability blob of size NxCxHxW and a label blog size of Nx1xHxC.
The code is based on softmax_loss_layer.hpp/softmax_loss_layer.cpp from https://github.com/TimoSaemann/caffe-segnet-cudnn5, which was a fork of BVLC/caffe.
This was a trial to implement a new layer in Caffe to be used in conjunction with Segnethttp://mi.eng.cam.ac.uk/projects/segnet/.
The original Multinomial Logistic Loss layer expects a label blob to have a size of Nx1x1x1 i.e. one label per one image. I would like to customise the probabilies obtained from the Softmax layer so I need a separate Multinomial Logistic Loss layer, and that the use of SoftmaxWithLoss layer is not possible.

Note: Only a CPU implementation is implemented.
