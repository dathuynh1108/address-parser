from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("artifacts")
model = AutoModelForTokenClassification.from_pretrained("artifacts")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# NER TESTING
test = "82 Đinh Bộ Lĩnh, Bình Thạnh, Thành phố Hồ Chí Minh"
ner_results = nlp(test)

def build_result(ner_results):
    entities = []
    current_entity = None

    for res in ner_results:
        word = res['word']
        label = res['entity'][2:]  # Remove B- or I-

        if res['entity'].startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'entity': label, 'text': word}
        elif res['entity'].startswith('I-') and current_entity and current_entity['entity'] == label:
            current_entity['text'] += ' ' + word
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

result = build_result(ner_results)
for entity in result:
    print(f"Entity: {entity['text']}, Type: {entity['entity']}")