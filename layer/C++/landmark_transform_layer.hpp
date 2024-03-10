#ifndef CAFFE_LANDMARK_TRANSFORM_LAYER_
#define CAFFE_LANDMARK_TRANSFORM_LAYER_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

/**
* Input: bottom[0]: Stage1 output landmarks [b, 136]
*        bottom[1]: Affine parameters [b, 6]
*
* Output: top[0]: Transformed landmarks [b, 136]
*/

template <typename Dtype>
class LandmarkTransformLayer : public Layer<Dtype> {
    public:
     explicit LandmarkTransformLayer(const LayerParameter& param)
         : Layer<Dtype>(param) {}
     virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
     virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

     virtual inline const char* type() const { return "LandmarkTransformLayer"; }
     virtual inline int ExactNumBottomBlobs() const {return 2;}
     virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
     virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {};
     virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom) {};
     virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom) {};

     bool inverse_;


};

}

#endif
