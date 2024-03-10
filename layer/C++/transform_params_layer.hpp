#ifndef CAFFE_TRANSFORM_PARAMS_LAYER_HPP_
#define CAFFE_TRANSFORM_PARAMS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe{

/**
 * input : bottom[0]  :Stage1 output landmarks[b, 136]
 * output: top[0] : Affine parameters[b, 6]
*/

template <typename Dtype>
class TransformParamsLayer : public Layer<Dtype> {
public:
    explicit TransformParamsLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "TransformParams"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){};
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom){};
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom){};

    Dtype* destinationx;


    int M_;
    int N_;

};

}

#endif
