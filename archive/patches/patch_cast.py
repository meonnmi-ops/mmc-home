#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# patch_cast.py — pointer declaration init အတွက် explicit cast ထည့်ရန်
# ပြဿနာ: nvcc က C++ အနေနဲ့ compile လုပ်လို့
#   double* hx = mmc_malloc(64);   <- C++ မှာ void* -> double* implicit မရ
# ပြင်ဆင်ချက်:
#   double* hx = (double*)(mmc_malloc(64));
# ပြင်မယ့်နေရာ: gen_node() ထဲက NK_VAR_DECL emission (init ရှိတဲ့အချက်)
# ============================================================
import io, sys

PATH = "src/compiler_v14.mmc"

OLD = 'result = mmc_string_concat(result, mmc_string_concat(" = ", gen_expr(init)))'

NEW = """သုံးသပ်ချက်အရ(mmc_string_length(node->type_str) > 0 နှင့် node->type_str[mmc_string_length(node->type_str) - 1] == '*') {
                result = mmc_string_concat(result, mmc_string_concat(" = (", mmc_string_concat(node->type_str, mmc_string_concat(")(", mmc_string_concat(gen_expr(init), "))")))))
            } မဟုတ်လျှင် {
                result = mmc_string_concat(result, mmc_string_concat(" = ", gen_expr(init)))
            }"""

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

count = src.count(OLD)
print("ရှာတွေ့သော အရေအတွက်: %d (၁ ခုတည်း ဖြစ်ရမယ်)" % count)
if count != 1:
    print("PATCH ABORT — နေရာ ထူးခြားမှု ရှိလို့ ဘာမှ မပြင်ရသေး")
    sys.exit(1)

src = src.replace(OLD, NEW)

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("PATCH OK — pointer declaration init များမှာ (TYPE*)(...) cast ထည့်ထုတ်နိုင်ပြီ")
