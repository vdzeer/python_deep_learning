import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# 1. Завантажуємо навчальні дані з CSV
data = pd.read_csv("terrain_dataset.csv") 
X = data[["R", "G", "B"]].values  # Ознаки (середні значення кольорів R, G, B)
y = data["Label"].values  # Мітки місцевості

# 2. Функція для навчання XGBoost класифікатора з параметрами
def train_classifier(X, y, n_estimators=100, max_depth=None, learning_rate=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return classifier

# 3. Функція для передбачення типів місцевості на новому зображенні
def predict_terrain(classifier, image, frame_size):
    h, w, _ = image.shape
    h = (h // frame_size) * frame_size
    w = (w // frame_size) * frame_size
    image = image[:h, :w] 

    # Розділяємо зображення на кадри і обчислюємо середні значення кольорів для кожного кадру
    frames = []
    for y in range(0, h, frame_size):
        for x in range(0, w, frame_size):
            frame = image[y:y+frame_size, x:x+frame_size]
            avg_color = np.mean(frame, axis=(0, 1))  # середнє значення R, G, B
            frames.append(avg_color)

    # Передбачаємо тип місцевості для кожного кадру
    frames = np.array(frames)
    predictions = classifier.predict(frames)
    
    # Формуємо сітку передбачень
    grid_h, grid_w = h // frame_size, w // frame_size
    prediction_grid = predictions.reshape(grid_h, grid_w)
    
    return prediction_grid

# 4. Функція для накладання передбачень на зображення
def overlay_predictions(image, prediction_grid, frame_size, output_path="output_image.jpg"):
    overlay_image = image.copy()
    grid_h, grid_w = prediction_grid.shape
    
    # Накладаємо мітки на зображення
    for i in range(grid_h):
        for j in range(grid_w):
            y, x = i * frame_size, j * frame_size
            label = prediction_grid[i, j]
            cv2.putText(
                overlay_image,
                str(label),
                (x + frame_size // 3, y + frame_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # Білий колір для тексту
                1,
                cv2.LINE_AA
            )
    
    # Відображення зображення для перевірки
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    # Зберігаємо зображення
    cv2.imwrite(output_path, overlay_image)
    print(f"Зображення збережено як {output_path}")

# Завантажуємо зображення для передбачення
image_path = 'input_image.jpg'

image = cv2.imread(image_path)
frame_size = 30  # Розмір кадру

# Навчаємо XGBoost класифікатор з налаштованою кількістю поколінь і глибиною
n_estimators = 10000  # Кількість дерев (поколінь)
max_depth = 100      # Максимальна глибина дерева
learning_rate = 0.1 # Швидкість навчання
classifier = train_classifier(X, y, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

# Передбачення типів місцевості на новому зображенні
terrain_prediction = predict_terrain(classifier, image, frame_size)
print("Сітка передбачень типів місцевості:\n", terrain_prediction)

# Накладання передбачень на зображення
overlay_predictions(image, terrain_prediction, frame_size)
