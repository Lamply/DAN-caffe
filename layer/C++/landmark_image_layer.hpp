#ifndef CAFFE_LANDMARK_IMAGE_LAYER_HPP_
#define CAFFE_LANDMARK_IMAGE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define CLIP(x, l, h) ((x) < (l) ? (l) : ((x) > (h) ? (h) : (x)))     


namespace caffe{
//  Input: bottom[0]: [b, 136] Transformed landmarks (flatten)
//         bottom[1]: [b, c, h, w] Source images
// 
//  Output: top[0]: Output images

template <typename Dtype>
class LandmarkImageLayer : public Layer<Dtype> {
    public:
     explicit LandmarkImageLayer(const LayerParameter& param)
         : Layer<Dtype>(param) {}
     virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
     virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top);

     virtual inline const char* type() const { return "LandmarkImage"; }
     virtual inline int ExactNumBottomBlobs() const { return 2;}
     virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
     virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {};
     virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_dowm,
                                const vector<Blob<Dtype>*>& bottom) {};
     virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_dowm,
                                const vector<Blob<Dtype>*>& bottom) {};
    
     int radius;
     int height;
     int weight;

};

}

#endif
