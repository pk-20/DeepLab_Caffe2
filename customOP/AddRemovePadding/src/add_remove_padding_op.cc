/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "add_remove_padding_op.h"

#include "caffe2/operators/conv_pool_op_base.h"
#include <vector>

#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)

void generatePoolImageChannel(const float* Xdata, float* Ydata, int height, int width, int pad_l, int pad_r, int pad_t, int pad_b)
{

  int outHeight = height + pad_t + pad_b;
  int outWidth = width + pad_l + pad_r;

  std::fill(Ydata, Ydata + outHeight * outWidth, 0);


  int xWS = 0, xWE = width - 1;
  int xHS = 0, xHE = height - 1;

  int yWS = 0, yWE = outWidth - 1;
  int yHS = 0, yHE = outHeight - 1;

  if(pad_l < 0)
  {
    xWS += -pad_l;
  }
  if(pad_r < 0)
  {
    xWE += pad_r;
  }
  if(pad_t < 0)
  {
    xHS += -pad_t;
  }
  if(pad_b < 0)
  {
    xHE += pad_b;
  }

  if(pad_l > 0)
  {
    yWS += pad_l; 
  }
  if(pad_r > 0)
  {
    yWE += -pad_r;
  }
  if(pad_t > 0)
  {
    yHS += pad_t;
  }
  if(pad_b > 0)
  {
    yHE += -pad_b;
  }







  for (int i = yHS, ix = xHS; i <= yHE; i++, ix++)
  {
    for (int j = yWS, jx = xWS; j <= yWE; j++, jx++)
    {

      Ydata[i*outWidth+j] = Xdata[ix*width+jx];
    }
  }

}



namespace caffe2 {
bool AddRemovePadding_OP<CPUContext>::RunOnDevice() {

  const auto& X = Input(0);
  CAFFE_ENFORCE(X.ndim() == 4);

  auto* Y = Output(0);

  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);

  int outChannels = channels;
  int outHeight = height + pad_t() + pad_b();
  int outWidth = width + pad_l() + pad_r();



  std::vector<int> output_dims;
  output_dims.push_back(X.dim32(0));
  output_dims.push_back(X.dim32(1));
  output_dims.push_back(outHeight);
  output_dims.push_back(outWidth);
  Y->Resize(output_dims);


  const float* Xdata = X.template data<float>(); 
  float* Ydata = Y->template mutable_data<float>();


  for (int n = 0; n < X.dim32(0); ++n) {

    for (int c = 0; c < channels; ++c) {
      
      generatePoolImageChannel(Xdata, Ydata, height, width, pad_l(), pad_r(), pad_t(), pad_b());

      
      // Do offset.
      Xdata += height * width;
      Ydata += outHeight * outWidth;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(AddRemovePadding, AddRemovePadding_OP<CPUContext>);

OPERATOR_SCHEMA(AddRemovePadding)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "1D input tensor")
    .Output(
        0,
        "Y",
        "1D output tensor");

} // namespace caffe2
