from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NlpHUST/electra-base-vn")

tokens = tokenizer(
    ["Việt", "Nam"],
    is_split_into_words=True,
    truncation=True,    
    padding=False,   
)

tokens_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
print(tokens_text)