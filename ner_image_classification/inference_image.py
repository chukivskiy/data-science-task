import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse

# Your class names (in English, in the same order as they appear in ImageFolder)
CLASS_NAMES = [
    'butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'
]


def load_model(model_path="best_resnet18_animals10.pth"):
    """
    Load the trained ResNet18 model.
    Note: weights=None because we use our own trained weights.
    """
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(CLASS_NAMES))

    # Load trained weights (map to CPU if GPU is not available)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_animal(image_path, model=None):
    """
    Predict the animal class from a single image.

    Args:
        image_path (str): Path to the image file
        model: Optional pre-loaded model (will be loaded if None)

    Returns:
        str: Predicted class name
    """
    if model is None:
        model = load_model()

    # Same transformations as used during validation/testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension [1, C, H, W]

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img = img.to(device)

    # Inference
    with torch.no_grad():
        output = model(img)
        pred_idx = torch.argmax(output, dim=1).item()

    predicted_class = CLASS_NAMES[pred_idx]
    return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal classification using trained ResNet18")
    parser.add_argument("--image", type=str, help="Path to the image (e.g. cat.jpg)")
    args = parser.parse_args()

    image_path = args.image

    # Interactive input if no argument provided
    if not image_path:
        image_path = input("Enter path to the image: ").strip()

    if not image_path:
        print("No image path provided!")
        exit(1)

    print(f"Loading model and predicting for: {image_path}")
    model = load_model()
    result = predict_animal(image_path, model)
    print(f"Predicted animal: {result}")