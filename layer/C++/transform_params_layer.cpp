#include <vector>
#include <stdio.h>

#include "caffe/layers/transform_params_layer.hpp"
#include "caffe/util/math_functions.hpp"

//Input : bottom[0] : Stage1 output landmarks[b, 136]
//Output : top[0] : Affine parameters[b, 6]

namespace caffe{

template <typename Dtype>
void TransformParamsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                       const vector<Blob<Dtype>*>& top){
    CHECK_EQ(bottom.size(),1) << "Require 1 inputs.";
    static Dtype dest[] ={28.0000012, 41.52647385, 28.18473534, 48.89341398, 29.00594527, 56.22849459, 30.54294376,
                   63.44273345, 33.39682386, 70.1509758, 37.81301279, 75.96648242, 43.24753411, 80.8196526,
                   49.22704001, 84.74777134, 55.99999891, 85.90433046, 62.77296086, 84.74777136, 68.75246374,
                   80.81965265, 74.18698812, 75.96648249, 78.60317555, 70.15097589, 81.45705415, 63.44273355,
                   82.99405419, 56.22849469, 83.81526719, 48.89341409, 84.00000136, 41.52647395, 33.20508288,
                   36.05814056, 36.70989054, 32.88769673, 41.65818614, 31.95748288, 46.76025352, 32.70177806,
                   51.53587937, 34.70101794, 60.46412169, 34.70101795, 65.23974756, 32.70177809, 70.34181493,
                   31.95748293, 75.29010749, 32.8876968, 78.79491666, 36.05814065, 55.999999, 40.50158622,
                   55.99999899, 45.29959221, 55.99999898, 50.06187029, 55.99999897, 54.97181038, 50.36656546,
                   58.21004389, 53.07869107, 59.19360786, 55.99999896, 60.06598282, 58.9213099, 59.19360787,
                   61.63343552, 58.21004391, 38.90346102, 41.06531973, 41.91323263, 39.29239318, 45.56029355,
                   39.34859745, 48.73460122, 41.80838019, 45.30799628, 42.4491319, 41.68195912, 42.39384911,
                   63.26539982, 41.80838022, 66.43970446, 39.34859749, 70.08676842, 39.29239324, 73.0965385,
                   41.06531979, 70.31803888, 42.39384916, 66.69200172, 42.44913194, 45.14172212, 67.10212643,
                   49.1410169, 65.52928901, 53.18444242, 64.65379684, 55.99999895, 65.38025847, 58.81555854,
                   64.65379685, 62.85898405, 65.52928904, 66.85828186, 67.10212647, 62.98245642, 70.94508574,
                   59.06514608, 72.62448044, 55.99999894, 72.94832477, 52.93485484, 72.62448043, 49.01754451,
                   70.94508572, 46.82482707, 67.32302035, 53.14206949, 67.0548055, 55.99999895, 67.36606539,
                   58.85793145, 67.05480551, 65.17517387, 67.32302038, 58.91110973, 69.0006451, 55.99999895,
                   69.3473227, 53.08889121, 69.00064509};
    destinationx = dest;
    int num = bottom[0]->shape(0);
    M_ = num;
    int count = bottom[0]->count(1);
    N_ = count;
    LOG_IF(INFO, Caffe::root_solver())
            << "num: " << num << " count: " << count;
}

template <typename Dtype>
void TransformParamsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                    const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape(2);
    top_shape[0] = M_;
    top_shape[1] = 6;
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void TransformParamsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
    Dtype destMean[2] = {(Dtype)0.};
    Dtype des_vec[N_];
    for (int n = 0; n < N_ / 2; n++){
        destMean[0] += destinationx[n * 2];
        destMean[1] += destinationx[n * 2 + 1]; 
    }
    destMean[0] /= (N_ / 2);
    destMean[1] /= (N_ / 2);          //destmean
    for (int n = 0; n < N_/2 ; n++){
        des_vec[n * 2] = destinationx[n * 2] - destMean[0];
        des_vec[n * 2 + 1] = destinationx[n * 2 + 1] - destMean[1];  //des_vec
    }

    for (int m = 0; m < M_; m++){
        const Dtype* source = bottom[0]->cpu_data() + m * N_;
        Dtype srcMean[2] = {(Dtype)0.};
        Dtype src_vec[N_], src_temp, sv_temp;
        for (int n = 0; n < N_ / 2; n++){
            srcMean[0] += source[n * 2];
            srcMean[1] += source[n * 2 + 1];
        }
        srcMean[0] /= (N_ / 2);
        srcMean[1] /= (N_ / 2);       //srcMean
        for (int n = 0; n < N_/2; n++){
            src_vec[n * 2] = source[n * 2] - srcMean[0];
            src_vec[n * 2 + 1] = source[n * 2 + 1] - srcMean[1]; //srcvec
        }
        src_temp = caffe_cpu_dot<Dtype>(N_, src_vec, src_vec);
        sv_temp = caffe_cpu_dot<Dtype>(N_, src_vec, des_vec);
        
        Dtype a = sv_temp / src_temp;
        Dtype b = (Dtype)0.;
        for (int n = 0; n < N_/2; n++){
            b += src_vec[n * 2] * des_vec[n * 2 + 1] - src_vec[n * 2 + 1] * des_vec[n * 2];
        }
        b /= src_temp;

        Dtype A[4] = {a, b, -b, a};
        Dtype srcMean2[2] = {(Dtype)0.};
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                1, 2, 2, (Dtype)1.,srcMean, A, (Dtype)0., srcMean2);
        caffe_copy(4, A, top[0]->mutable_cpu_data() + m * 6);
        Dtype bias[2] = {destMean[0] - srcMean2[0], destMean[1] - srcMean2[1]};
        caffe_copy(2, bias, top[0]->mutable_cpu_data() + m * 6 + 4);
    }
}

#ifdef CPU_ONLY
STUB_GPU(TransformParamsLayer);
#endif

INSTANTIATE_CLASS(TransformParamsLayer);
REGISTER_LAYER_CLASS(TransformParams);

}
