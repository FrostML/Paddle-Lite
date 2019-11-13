// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/x86/cast_compute.h"

typedef paddle::lite::kernels::x86::CastCompute<PRECISION(kFloat), float>
    CastFp32;
typedef paddle::lite::kernels::x86::CastCompute<PRECISION(kFloat), int32_t>
    CastInt32;
typedef paddle::lite::kernels::x86::CastCompute<PRECISION(kFloat), int64_t>
    CastInt64;

REGISTER_LITE_KERNEL(cast, kX86, kFloat, kNCHW, CastFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(cast, kX86, kInt32, kNCHW, CastInt32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(cast, kX86, kInt64, kNCHW, CastInt64, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
