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

#ifndef ADD_REMOVE_OP_H_
#define ADD_REMOVE_OP_H_


#include "caffe2/core/common_omp.h"

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {


template<class Context>
class AddRemovePadding_OP final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  AddRemovePadding_OP(const OperatorDef& operator_def, Workspace* ws) : ConvPoolOpBase<Context>(operator_def, ws) {}

  bool RunOnDevice() override;
};

// CPU operators
template<>
class AddRemovePadding_OP<CPUContext> final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  AddRemovePadding_OP(const OperatorDef& operator_def, Workspace* ws) : ConvPoolOpBase<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override;
};
} // namespace caffe2

#endif