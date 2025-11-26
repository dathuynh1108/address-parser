from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import time
tokenizer = AutoTokenizer.from_pretrained("dathuynh1108/ner-address-electra-base-vn")
model = AutoModelForTokenClassification.from_pretrained("dathuynh1108/ner-address-electra-base-vn")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# NER TESTING
test = "số 79 đường 339, Phường Phước Long B, Quận 9, Thành phố Hồ Chí Minh, Việt Nam"
start_time = time.time()
ner_results = nlp(test)
print(f"NER processing time: {time.time() - start_time} seconds")

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