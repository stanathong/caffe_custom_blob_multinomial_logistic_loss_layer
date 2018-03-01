#include <algorithm>
#include <cfloat> // Add following SoftmaxWithLoss
#include <cmath>
#include <vector>


#include "caffe/layer.hpp"	// Add following SoftmaxWithLoss
#include "caffe/layers/custom_blob_multinomial_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CustomBlobMultinomialLogisticLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // Validate the shape of prob blob and label blob
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) 
  {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  weight_by_label_freqs_ =
    this->layer_param_.loss_param().weight_by_label_freqs();
  
  if (weight_by_label_freqs_) 
  {
    vector<int> count_shape(1, this->layer_param_.loss_param().class_weighting_size());
    label_counts_.Reshape(count_shape);
    CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(), bottom[0]->channels())
		<< "Number of class weight values does not match the number of classes.";
    float* label_count_data = label_counts_.mutable_cpu_data();
    for (int i = 0; i < this->layer_param_.loss_param().class_weighting_size(); i++) 
	{
        label_count_data[i] = this->layer_param_.loss_param().class_weighting(i);
    }
  }
}

template <typename Dtype>
void CustomBlobMultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::Reshape(bottom, top);

  softmax_axis_ = 
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  CHECK_EQ(softmax_axis_, 1) << "The probability axis must be the same as channel axis." ;

  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);

  if (top.size() >= 2) 
  {
    top[1]->ReshapeLike(*bottom[0]);
  }
  if (weight_by_label_freqs_) 
  {
    CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(), bottom[0]->channels())
		<< "Number of class weight values does not match the number of classes.";
  }
}

template <typename Dtype>
void CustomBlobMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  //int num = bottom[0]->num();
  int nlabel = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
//  for (int i = 0; i < num; ++i) {
//    int label = static_cast<int>(bottom_label[i]);
//    Dtype prob = std::max(
//        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
//    loss -= log(prob);
//  }
//  top[0]->mutable_cpu_data()[0] = loss / num;

  for (int i = 0; i < outer_num_; ++i)	// N
  {
    for (int j = 0; j < inner_num_; j++) // HxW
	{
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) 
	  {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, nlabel);
      const int idx = i * dim + label_value * inner_num_ + j;
      if (weight_by_label_freqs_) 
	  {
        const float* label_count_data = label_counts_.cpu_data();
        loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN)))
            * static_cast<Dtype>(label_count_data[label_value]);
      } else 
	  {
        loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN)));
      }
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void CustomBlobMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
/*
  if (propagate_down[1]) 
  {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) 
  {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + label] = scale / prob;
    }
  }
*/
  if (propagate_down[1]) 
  {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) 
  {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), prob_data, bottom_diff);	
			// copy probabilities to bottom_diff for NxCxHxW
			// bottom_diff = S
    const Dtype* label = bottom[1]->cpu_data();
    int dim = bottom[0]->count() / outer_num_; // NxHxW
    int count = 0;
    const float* label_count_data = 
        weight_by_label_freqs_ ? label_counts_.cpu_data() : NULL;
    for (int i = 0; i < outer_num_; ++i) // N
	{
      for (int j = 0; j < inner_num_; ++j) // HxW
	  {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) 
		{
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) 
		  {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } 
		else 
		{
          const int idx = i * dim + label_value * inner_num_ + j;
		  // diff for correct label = S-1
		  bottom_diff[idx] -= 1;

          if (weight_by_label_freqs_) 
		  {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) 
			{
              bottom_diff[i * dim + c * inner_num_ + j] *= static_cast<Dtype>(label_count_data[label_value]);
            }
          }
          ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CustomBlobMultinomialLogisticLossLayer);
#endif

INSTANTIATE_CLASS(CustomBlobMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(CustomBlobMultinomialLogisticLoss);

}  // namespace caffe
