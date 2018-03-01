#ifndef CAFFE_CUSTOM_BLOB_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
#define CAFFE_CUSTOM_BLOB_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/* 
 * Description: Computes loss based on multinomial logistic loss
 *		The code is adapted from multinomial_logistic_loss_layer and
 *		softmax_loss_layer.
 * Input: bottom input Blob vector (length 2)	
 *	1) the probabilities from softmax
 *			N x #grade x H x W
 *	2) label of size N x 1 x H x W
 * Output: top output Blob vector (length 1)
 *	loss: 1x1x1x1
 */
template <typename Dtype>
class CustomBlobMultinomialLogisticLossLayer : public LossLayer<Dtype> {
 public:
  explicit CustomBlobMultinomialLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  // Added LayerSetUp as it exists in SoftmaxWithLossLayer but not in MultinomialLogLoss
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CustomBlobMultinomialLogisticLoss"; }

  // Added following SoftmaxWithLossLayer
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Added following SoftmaxWithLossLayer
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the multinomial logistic loss error gradient w.r.t. the
   *        predictions.
   *
   * Gradients cannot be computed with respect to the label inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   *      propagate_down[1] must be false as we can't compute gradients with
   *      respect to the labels.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ \hat{p} @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial \hat{p}} @f$
   *   -# @f$ (N \times 1 \times H \times W) @f$ // modify label to be of size Nx1xHxC 
   *	  instead of Nx1x1x1
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // Added following SoftmaxWithLossLayer
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Add the member variables according to SoftmaxWithLossLayer

  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Whether to normalize the loss by the total number of values present
  /// (otherwise just by the batch size).
  bool normalize_;
  /// Whether to weight labels by their batch frequencies when calculating
  /// the loss
  bool weight_by_label_freqs_;
  Blob<float> label_counts_; // For Blob data, before use, must get data from memory either const or mutable.

  int softmax_axis_, outer_num_, inner_num_;
};

}  // namespace caffe

#endif  // CAFFE_CUSTOM_BLOB_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
