from transformers import pipeline

# Load vision pipeline with specific model
vision_classifier = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224"
)

try:
    # Process image
    result = vision_classifier("./ficheiros/prado.jpg")

    # Format and display results
    print("\nImage Classification Results:")
    for pred in result:
        print(f"Label: {pred['label']:<20} Confidence: {pred['score']:.2%}")

except Exception as e:
    print(f"Error: {e}")
