import argparse
from transformers import pipeline
from inference_image import predict_animal, load_model  # import the prediction function

# Load the NER model once at startup
ner = pipeline(
    "ner",
    model="ner_animal_final",
    tokenizer="ner_animal_final",
    aggregation_strategy="simple"
)

# Load the image classification model once
image_model = load_model("best_resnet18_animals10.pth")


def check(text: str, image_path: str) -> bool:
    """
    Check if the animal mentioned in the text matches the animal predicted from the image.

    Args:
        text (str): Text description that should contain an animal name
        image_path (str): Path to the image file

    Returns:
        bool: True if animals match (considering negation), False otherwise
    """
    # 1. Extract animal from text using NER
    entities = ner(text)
    extracted = None

    for entity in entities:
        if entity['entity_group'] == 'ANIMAL':
            extracted = entity['word'].lower().strip()
            break

    if not extracted:
        print(f"No animal found in the text: {text}")
        return False

    # 2. Predict animal from the image
    try:
        predicted = predict_animal(image_path, model=image_model).lower().strip()
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

    # 3. Compare the results
    print(f"Text     → {extracted}")
    print(f"Image    → {predicted}")

    is_match = extracted == predicted

    # # Handle negation in the text
    # text_lower = text.lower()
    # negation_words = ["not", "no", "isn't", "isnt", "doesn't", "dont", "never", "n't"]
    # has_negation = any(word in text_lower for word in negation_words)
    #
    # if has_negation:
    #     print("Negation detected → inverting the result")
    #     is_match = not is_match

    print(f"Result: {is_match}")
    return is_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check whether the animal in the text matches the animal in the image"
    )
    parser.add_argument("--text", type=str,
                        help="Text description (e.g. 'There is a cow in the picture.')")
    parser.add_argument("--image", type=str,
                        help="Path to the image file (e.g. cat.jpg)")
    args = parser.parse_args()

    # If arguments are not provided → ask interactively
    text = args.text
    image_path = args.image

    if not text:
        text = input("Enter text description: ").strip()

    if not image_path:
        image_path = input("Enter path to the image: ").strip()

    # Basic validation
    if not text or not image_path:
        print("Text or image path not provided. Exiting.")
        exit(1)

    # Run the check
    result = check(text, image_path)
    print(f"\nFinal result: {result}")