#ifndef CAFFE_AFFINE_TRANSFORM_LAYER_HPP_
#define CAFFE_AFFINE_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** Input: bottom[0]: Source images
*          bottom[1]: Affine parameters
*
*   Output: top[0]: Transformed images
*/
template <typename Dtype>
class AffineTransformLayer : public Layer<Dtype> {
    public:
     explicit AffineTransformLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
     virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
     virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

     virtual inline const char* type() const { return "AffineTransform"; }
     virtual inline int ExactNumBottomBlobs() const {return 2; }
     virtual inline int ExactNumTopBlobs() const {return 1; }

    protected:
     virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){};
     virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                               const vector<bool>& propagate_down, 
                               const vector<Blob<Dtype>*>& bottom){};
     virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                               const vector<bool>& propagate_down, 
                               const vector<Blob<Dtype>*>& bottom){};
    

     int num;
     int weight;
     int height;
     int count;
         
};

}

#endif
