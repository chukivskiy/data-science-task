from transformers import pipeline

# Load pretrained model
ner_pipeline = pipeline(
    "ner",
    model="ner_animal_final",          # model folder
    tokenizer="ner_animal_final",
    aggregation_strategy="simple"
)


test_sentences = [
    "There is a cow standing in the field.",
    "Look at this cute squirrel eating nuts!",
    "Is that a dog or a fox?",
    "I see a beautiful butterfly on the flower.",
    "This picture shows a horse running fast.",
    "There is no cat here, it's just a shadow.",
    "The photo features a spider on the wall.",
    "Wow, it's a chicken crossing the road!",
    "I think there is an elephant in the background.",
    "This is definitely not a sheep, it's a goat.",
    # негативні приклади
    "It's not cat",
    "There is a beautiful sunset in the picture.",
    "Look at this amazing mountain view!",
    "Is that a car in the photo?",
    "The main object is a smartphone.",
    "I see a red house here."
]

print("\nTesting:\n" + "="*50)

for sentence in test_sentences:
    result = ner_pipeline(sentence)
    print(f"\nSentence: {sentence}")
    if result:
        for entity in result:
            if entity['entity_group'] == 'ANIMAL':
                print(f"  → Animal found: {entity['word']} (score: {entity['score']:.4f})")
    else:
        print("  → No animal found")
    print("-" * 50)