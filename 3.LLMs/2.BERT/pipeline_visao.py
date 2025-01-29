from transformers import pipeline

# carregar o pipeline de visão
vision_classifier = pipeline(task="image_classification")

result = vision_classifier("https://www.example.com/image.jpg")