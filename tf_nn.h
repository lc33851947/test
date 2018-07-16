#ifndef __TF_NN_H__
#define __TF_NN_H__

conv_layer tf_nn_conv2d(conv_layer input_tensor, conv_kernel kernel_tensor, int strides, int padding);
conv_layer tf_nn_pooling(conv_layer input_tensor, int ksize, int strides, int padding,char mode);
void tf_nn_relu_conv(conv_layer input_tensor);
void tf_nn_add(conv_layer conv ,float* bias);
void tf_nn_softmax(fc_layer h_fc2, float *out);
void tf_nn_BatchNormalize(conv_layer input_tensor);

#endif