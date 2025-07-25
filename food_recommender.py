def recommend_food(emotion):
    food_map = {
        "happy": "Pizza",
        "sad": "Soup",
        "angry": "Steak",
        "surprised": "Sushi"
    }
    return food_map.get(emotion, "Unknown Food")

