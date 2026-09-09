"""Fixed evaluation texts used by every capture and gate.

Kept tiny and explicit so a golden capture is reproducible and reviewable.
The Arabic lines are Modern Standard / Gulf-flavoured everyday utterances;
the mixed lines embed English brand/technical tokens inside Arabic sentences,
which is the realistic code-switching case for the target deployment.
"""

CASES = [
    # (case_id, language_id, text)
    ("en_short", "en", "Hello."),
    ("en_plain", "en", "Hello, this is the reference implementation speaking."),
    ("en_numbers", "en", "Your appointment is on March 3rd at 9:45 AM, and the total is $124.50."),
    ("en_long", "en",
     "The quick brown fox jumps over the lazy dog, and then it turns around and does it "
     "again because the sentence needs to be long enough to exercise a real decode loop."),
    ("ar_short", "ar", "مرحبا."),
    ("ar_plain", "ar", "حياك الله، موعدك بكرة الساعة التاسعة صباحًا."),
    ("ar_numbers", "ar", "رقم الطلب مية وأربعة وعشرين، والمبلغ ثلاثمئة ريال."),
    ("ar_long", "ar",
     "أهلاً وسهلاً بك في خدمة العملاء، نحن سعداء بتواصلك معنا اليوم، "
     "وسوف نقوم بمراجعة طلبك والرد عليك في أقرب وقت ممكن."),
    ("mixed_ar_en", "ar", "حياك الله، حسابك على Netflix تم تجديده اليوم."),
    ("mixed_en_ar", "en", "Please say السلام عليكم before you start the call."),
    ("punct_edge", "en", "Wait... what?! - okay; fine: let's go."),
    ("ws_edge", "en", "   spaced    out   text   "),
]

CASES_BY_ID = {c[0]: c for c in CASES}
