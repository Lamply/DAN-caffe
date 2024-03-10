#include <vector>

#include "caffe/layers/landmark_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
template <typename Dtype>
    void LandmarkTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top){
        CHECK_EQ(bottom.size(),2)<<"Need 2 inputs.";
        inverse_ = this->layer_param_.landmark_transform_param().inverse();
	}

template <typename Dtype>
    void LandmarkTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    											const vector<Blob<Dtype>*>& top){
    	top[0]->ReshapeLike(*bottom[0]);
    }

template <typename Dtype>
    void LandmarkTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    												const vector<Blob<Dtype>*>& top){
		/**
		* Input: bottom[0]: Stage1 output landmarks [b, 136]
		*        bottom[1]: Affine parameters [b, 6]
		*
		* Output: top[0]: Transformed landmarks [b, 136]
		*/
    	Dtype A[4];
    	Dtype bias[2];
    	int count = bottom[0]->count(1);
    	for (int n = 0; n < bottom[0]->shape(0); n++){
    		caffe_copy<Dtype>(4, bottom[1]->cpu_data() + n * 6, A);
    		caffe_copy<Dtype>(2, bottom[1]->cpu_data() + n * 6 + 4, bias);
    		if (inverse_) {
    			Dtype A_inv[4];
    			Dtype A_val = A[0] * A[3] - A[1] * A[2];
    			CHECK_NE(A_val, (Dtype)0.)<<"Value of matrix A is 0!";
    			A_inv[0] = A[3] / A_val;
    			A_inv[1] = -A[1] / A_val;
    			A_inv[2] = -A[2] / A_val;
    			A_inv[3] = A[0] / A_val;
				Dtype bias1[2] = {(Dtype)0.};
            	caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1, 2, 2, (Dtype)1.,
           				bias, A_inv, (Dtype)0., bias1);
            	bias[0] = -bias1[0];
            	bias[1] = -bias1[1];
            	caffe_copy<Dtype>(4, A_inv, A);
			}

            Dtype transformed_landmark[count];
            for (int i = 0; i < count / 2; i++){
            	caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1 ,2, 2, (Dtype)1., 
            		bottom[0]->cpu_data() + n * count + i * 2, A, (Dtype)0., 
            		transformed_landmark +i * 2);
            	transformed_landmark[i * 2] += bias[0];
            	transformed_landmark[i * 2 + 1] += bias[1];
            caffe_copy<Dtype>(count, transformed_landmark, top[0]->mutable_cpu_data() + n * count);

            }
    	}
    }
     
#ifdef CPU_ONLY
STUB_GPU(LandmarkTransformLayer);
#endif

INSTANTIATE_CLASS(LandmarkTransformLayer);
REGISTER_LAYER_CLASS(LandmarkTransform);   
}
