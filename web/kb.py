"""Knowledge base — easy to edit, no rebuild needed."""

KB = [
    # Format: (keyword pattern, answer)
    # Python KB — chat_v14 ကို မခေါ်ဘဲ ချက်ချင်း ဖြေ
    ("MMC ဆိုတာ", "MMC (Myanmar Programming Language) က မြန်မာစာနဲ့ ရေးတဲ့ programming language ပါ။ 100% Pure MMC — No Python, No C ။"),
    ("v14", "v14 က MMC compiler ရဲ့ နောက်ဆုံး version ပါ။ Chat Engine အပြည့်အစုံ အလုပ်လုပ်တယ် ။"),
    ("ဘယ်လို ရေးထားလဲ", "ကျွန်တော်ကို MMC နဲ့ ရေးထားတယ် ။ Python မပါ ၊ C မပါ ။ Compiler ကနေ C ကို ထုတ်ပြီး gcc နဲ့ ပြန်စုတယ် ။"),
    ("မြန်မာနိုင်ငံ", "မြန်မာနိုင်ငံ က အရှေ့တောင်အာရှ မှာ ရှိတဲ့ နိုင်ငံ ပါ ။ မြို့တော် က နေပြည်တော် ။ လူဦးရေ ~54 သန်း ။"),
    ("ရာသီဥတု", "မြန်မာနိုင်ငံ က အပူပိုင်း ရာသီဥတု ရှိတယ် ။ သုံးရာသီ — နွေ ၊ မိုး ၊ ဆောင်း ။"),
    ("မင်္ဂလာနံနက်ခင်း", "နံနက်ခင်း မင်္ဂလာပါ ။ ဒီနေ့ ဘာလုပ်ဖို့ ရှိလဲ?"),
    ("အသက်", "ကျွန်တော်က AI ပါ ။ အသက် ဆိုတာ မရှိဘူး ။ 2026 မှာ ဖန်တီးခဲ့တယ် ။"),
    ("compiler", "MMC Compiler က မြန်မာစာကို C ကို ပြောင်းပေးတယ် ။ ပြီးရင် gcc နဲ့ binary ထုတ်တယ် ။"),
    ("chat engine", "Chat engine က GGUF model ကို ဖတ်ပြီး ၊ prompt ကို tokenize လုပ်ပြီး ၊ token တွေ generate လုပ်တယ် ။"),
    ("rule", "Rule layer က common questions တွေကို ချက်ချင်း ဖြေတယ် ။ Model ကို မခေါ်ဘဲ ။"),
]


def match_kb(text):
    """Find answer from KB by keyword matching."""
    text_lower = text.lower().strip()
    for pattern, answer in KB:
        if pattern.lower() in text_lower:
            return answer
    return None
