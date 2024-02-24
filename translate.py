from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

hi_text = "ma dat do ass awer een starkt steck"
chinese_text = "hin jong"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate luxemburgish to French
tokenizer.src_lang = "lb"
encoded_hi = tokenizer(hi_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("fr"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "La vie est comme une boîte de chocolat."

# translate luxemburgish to german
tokenizer.src_lang = "lb"
encoded_zh = tokenizer(hi_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("de"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Life is like a box of chocolate."