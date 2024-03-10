#include <vector>


#include "caffe/layers/affine_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define CLIP(x, l, h) ((x) < (l) ? (l) : ((x) > (h) ? (h) : (x)))  

namespace caffe{

template <typename Dtype>
void AffineTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                             const vector<Blob<Dtype>*>& top){
    CHECK_EQ(bottom.size(), 2)<< "Require 2 inputs.";
    num = bottom[0]->shape(0);
    height = bottom[0]->shape(2);
    weight = bottom[0]->shape(3);
    count = bottom[0]->count(1);
}

template <typename Dtype>
void AffineTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top){
    vector<int> shape = bottom[0]->shape();
    shape[1] = 1;
    top[0]->Reshape(shape);
}

template <typename Dtype>
void AffineTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
    for (int n=0; n < num; n++){
        Dtype A[4];
        Dtype bias[2];
        caffe_copy<Dtype>(4, bottom[1]->cpu_data() + n * 6, A);
        caffe_copy<Dtype>(2, bottom[1]->cpu_data() + n*6+4, bias);      //A, bias

        Dtype A_inv[4];
        Dtype A_val = A[0] * A[3] - A[1] * A[2];
        CHECK_NE(A_val, (Dtype)0.)<<"Value of matrix A is 0!";
        A_inv[0] = A[3] / A_val;
        A_inv[1] = -A[1] / A_val;
        A_inv[2] = -A[2] / A_val;
        A_inv[3] = A[0] / A_val;
        Dtype bias1[2];
        caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1, 2, 2, (Dtype)1.,
           bias, A_inv, (Dtype)0., bias1);
        bias[0] = -bias1[0]; 
        bias[1] = -bias1[1];                           //A_inv, new bias

        Dtype pixels[weight * height][2] ={(Dtype)0.};
        Dtype outPixels[weight * height][2]={(Dtype)0.};   
        for (int y = 0; y < height; y++){
            for (int x = 0; x < weight; x++){
                pixels[y * weight + x][0] = (Dtype)x;
                pixels[y * weight + x][1] = (Dtype)y;

                Dtype outPixels1[2] = {(Dtype)0.};
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 2, 2, (Dtype)1.,
                    pixels[y * weight + x], A_inv, (Dtype)0., outPixels1); 
                outPixels[y * weight + x][0] = CLIP(outPixels1[0] + bias[0], 0, weight -2);
                outPixels[y * weight + x][1] = CLIP(outPixels1[1] + bias[1], 0, height -2);     
            }
        }                                             // pixels  outPixels

        int outPixelsMinMin[weight * height][2];
        int outPixelsMaxMin[weight * height][2];
        int outPixelsMinMax[weight * height][2];
        int outPixelsMaxMax[weight * height][2];

        Dtype dx[weight * height];
        Dtype dy[weight * height];

        Dtype outImg[weight * height] = {(Dtype)0.};
        const Dtype* img = bottom[0]->cpu_data() + n * count;
        
        for (int y = 0; y < height; y++){
            for (int x = 0; x < weight; x++){
                outPixelsMinMin[y * weight + x][0] = (int) outPixels[y * weight + x][0];
                outPixelsMinMin[y * weight + x][1] = (int) outPixels[y * weight + x][1];
                outPixelsMaxMin[y * weight + x][0] = (int) outPixels[y * weight + x][0] + 1;
                outPixelsMaxMin[y * weight + x][1] = (int) outPixels[y * weight + x][1];
                outPixelsMinMax[y * weight + x][0] = (int) outPixels[y * weight + x][0];
                outPixelsMinMax[y * weight + x][1] = (int) outPixels[y * weight + x][1] + 1;
                outPixelsMaxMax[y * weight + x][0] = (int) outPixels[y * weight + x][0] + 1;
                outPixelsMaxMax[y * weight + x][1] = (int) outPixels[y * weight + x][1] + 1;

                dx[y * weight + x] = outPixels[y * weight + x][0] - outPixelsMinMin[y * weight + x][0];
                dy[y * weight + x] = outPixels[y * weight + x][1] - outPixelsMinMin[y * weight + x][1];

                outImg[y * weight + x] += (1 -dx[y * weight + x]) * (1 - dy[y * weight + x]) *
                    img[outPixelsMinMin[y * weight + x][1] * weight +outPixelsMinMin[y * weight +x][0]];
                outImg[y * weight + x] += dx[y * weight + x] * (1 - dy[y * weight + x]) *
                    img[outPixelsMaxMin[y * weight + x][1] * weight +outPixelsMaxMin[y * weight +x][0]];
                outImg[y * weight + x] += (1 -dx[y * weight + x]) * dy[y * weight + x] *
                    img[outPixelsMinMax[y * weight + x][1] * weight +outPixelsMinMax[y * weight +x][0]];
                outImg[y * weight + x] += dx[y * weight + x] * dy[y * weight + x] *
                    img[outPixelsMaxMax[y * weight + x][1] * weight +outPixelsMaxMax[y * weight +x][0]];
            }
        }                   //outPixelsMaxMax, outPixelsMaxMin, outPixelsMinMax,outPixelsMinMin, dx, dy


        caffe_copy<Dtype>(weight * height, outImg, top[0]->mutable_cpu_data() + n * weight * height);

    }
}

#ifdef CPU_ONLY
STUB_GPU(AffineTransformLayer);
#endif

INSTANTIATE_CLASS(AffineTransformLayer);
REGISTER_LAYER_CLASS(AffineTransform);

}
