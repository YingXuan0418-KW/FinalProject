from ultralytics import YOLO

# Load models
emotion_model = YOLO('/home/nvidia04/FinalProject/yolov8n-cls.pt')



food_model = YOLO('/home/nvidia04/FinalProject/food_model/food.pt')

# Define mapping from emotion to food cuisine suggestion
emotion_to_food = {
    'happy': 'Ice Cream',
    'sad': 'Comfort Food',
    'angry': 'Spicy Food',
    'neutral': 'Salad',
    # ... add more mappings based on your emotion labels and food types
}

def detect_emotion(image_path):
    results = emotion_model(image_path)
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        emotion_label = boxes.cls[0].item()
        emotion_name = emotion_model.names[int(emotion_label)]
    else:
        emotion_name = 'unknown'
    
    return emotion_name


def suggest_food(emotion):
    return emotion_to_food.get(emotion, 'No suggestion')

def main(image_path):
    emotion = detect_emotion(image_path)
    print(f"Detected emotion: {emotion}")
    
    suggestion = suggest_food(emotion)
    print(f"Suggested food based on emotion: {suggestion}")

    # Optionally, if you want to validate or recognize food cuisine from an image:
    # food_results = food_model(image_path)
    # process food_results ...

if __name__ == "__main__":
    test_image = '/home/nvidia04/FinalProject/emotion_model/dataset/test/0a29ef6f5e1110b20073182cbc98fab0c95d502920b6b6b2bbdf27cc~12fffff.jpg'
    main(test_image)
