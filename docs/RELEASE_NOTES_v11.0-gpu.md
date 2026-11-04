# MMC v11.0-gpu 🚀

**No Fake No Simulate — Pure MMC**

မြန်မာစာ keyword တွေနဲ့ ရေးထားတဲ့ programming language ကနေ တိုက်ရိုက် GPU kernel တွေ ရေးပြီး ပြေးလို့ရပြီ —

## ⚡ GPU kernels (GTX 1660 Ti / sm_75 ပေါ်မှာ တကယ်ပြေးပြီး)

| Kernel | အရွယ်အစား | ရလဒ် |
|---|---|---|
| GELU | 10,000,000 | GPU 8.6ms vs CPU 406ms → **47.4x မြန်** |
| LayerNorm | 50,000 × 256 | GPU 43.8ms vs CPU 228ms → **5.2x မြန်** |
| Softmax | 64 × 64 | GPU = CPU ✅ |
| Matmul | 64 × 64 × 64 | GPU = CPU ✅ |
| ReLU | 1,000,000 | GPU = CPU ✅ |
| Transpose | 512 × 512 | GPU = CPU ✅ |
| Attention score | 64 × 64 (d=16) | GPU = CPU ✅ |

အားလုံးက `.mmc` ဖိုင်သက်သက် — ကမ္ဘာလုံး (global kernel) keyword နဲ့ ရေးထားတာ။
Fake မဟုတ်၊ Simulate မဟုတ် — CUDA ပေါ်မှာ တကယ်ပြေးတယ်။

## 🧠 Self-hosting
- Compiler ကို MMC နဲ့ပဲ ရေးထားတယ်
- Fixpoint verified: `bf78d0515e0c429083a03d16ea46a55b7d36f2e811455b9aa6bfae277df3fde8`

## 📖 ဘယ်လို စမ်းကြည့်မလဲ
README ထဲက **🚀 GPU လမ်းညွှန်** ကဏ္ဍကို ဖတ်ပါ — မြန်မာလို အဆင့်လိုက် ရှင်းပြထားတယ်။

## 🙏 Credits
Myanmar developer community အတွက် ချစ်ခြင်းနဲ့ တည်ဆောက်ထားပါတယ်။ 🇲🇲
