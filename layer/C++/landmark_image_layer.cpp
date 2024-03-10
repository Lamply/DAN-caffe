#include <vector>
#include <math.h>

#include "caffe/layers/landmark_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void LandmarkImageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                           const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom.size(), 2)<<"Need 2 inputs.";
    radius = this->layer_param_.landmark_image_param().radius();
    height = bottom[1]->shape(2);
    weight = bottom[1]->shape(3);
}

template <typename Dtype>
void LandmarkImageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    vector<int> out_shape = bottom[1]->shape();
    out_shape[1] = 1;
    top[0]->Reshape(out_shape);
}

template <typename Dtype>
void LandmarkImageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
    int count = bottom[1]->count(2);
    int num_landmarks = bottom[0]->count(1) / 2;
    for (int n =0 ; n < bottom[0]->shape(0); n++){
        Dtype img[count] = {(Dtype)0.};
        const Dtype* landmarks  = bottom[0]->cpu_data() + n * 2 * num_landmarks;  //landmarks [136,]
        for (int i = 0; i < num_landmarks; i++){
            Dtype landmark[2];
            landmark[0] = landmarks[i * 2];
            landmark[1] = landmarks[i * 2 + 1];
            landmark[0] = CLIP(landmark[0], radius, weight - 1 - radius);
            landmark[1] = CLIP(landmark[1], radius, height - 1 - radius);
            int intlandmark[2];
            intlandmark[0] = (int)landmark[0];
            intlandmark[1] = (int)landmark[1];
            for (int y = intlandmark[1] - radius; y <= intlandmark[1] + radius; y++){
                for (int x = intlandmark[0] - radius; x <= intlandmark[0] + radius; x++){
                    Dtype distance[2];
                    distance[0] = x - landmark[0];
                    distance[1] = y - landmark[1];
                    Dtype l2_distance = sqrt(distance[0] * distance[0] + distance[1] * distance[1]);
                    Dtype val = 1 / (1 + l2_distance);
                    if (val > img[y * weight + x]){
                        img[y * weight + x] = val;
                    }
                }
            }
        }
        caffe_copy(count, img, top[0]->mutable_cpu_data() + n * count);

    }
}

#ifdef CPU_ONLY
STUB_GPU(LandmarkImageLayer);
#endif

INSTANTIATE_CLASS(LandmarkImageLayer);
REGISTER_LAYER_CLASS(LandmarkImage);

}
