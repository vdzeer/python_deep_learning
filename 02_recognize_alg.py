import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Завантаження датасету та вилучення RGB ознак
data = pd.read_csv("terrain_dataset.csv")
X = data[["R", "G", "B"]].values
y = data["Label"].values

# 2. Навчання XGBoost класифікатора
def train_classifier(X, y, n_estimators=100, max_depth=None, learning_rate=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print("Звіт класифікації:\n", classification_report(y_test, y_pred))
    print("Точність:", accuracy_score(y_test, y_pred))
    
    return classifier

# 3. Прогноз місцевості з оптимізацією толерантності
def predict_terrain_with_tolerance(classifier, image, frame_size, tolerance=0.2):
    h, w, _ = image.shape
    h = (h // frame_size) * frame_size
    w = (w // frame_size) * frame_size
    image = image[:h, :w]

    frames = []
    for y in range(0, h, frame_size):
        for x in range(0, w, frame_size):
            frame = image[y:y+frame_size, x:x+frame_size]
            avg_color = np.mean(frame, axis=(0, 1))  # Середні значення RGB
            frames.append(avg_color)

    frames = np.array(frames)
    predictions = classifier.predict_proba(frames)

    optimized_predictions = []
    for pred in predictions:
        max_prob = np.max(pred)
        best_class = np.argmax(pred)
        
        if max_prob < 1 - tolerance:
            possible_classes = np.where(pred >= max_prob - tolerance)[0]
            best_class = possible_classes[np.argmax(pred[possible_classes])]
        
        optimized_predictions.append(best_class)

    optimized_predictions = np.array(optimized_predictions)
    grid_h, grid_w = h // frame_size, w // frame_size
    prediction_grid = optimized_predictions.reshape(grid_h, grid_w)
    
    return prediction_grid

# 4. Накладення прогнозів на зображення
def overlay_predictions(image, prediction_grid, frame_size, output_path="output_image.jpg"):
    overlay_image = image.copy()
    grid_h, grid_w = prediction_grid.shape
    
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
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    cv2.imwrite(output_path, overlay_image)
    print(f"Зображення збережено як {output_path}")

# 5. Побудова розподілів впевненості класів
def plot_class_confidences(predictions, tolerance, save_path="class_confidence_distribution.png"):
    confidence_df = pd.DataFrame(predictions, columns=["Forest", "Field", "Road"])
    for column in confidence_df.columns:
        sns.histplot(confidence_df[column], kde=True, label=column, bins=20)
    plt.axvline(1 - tolerance, color="red", linestyle="--", label=f"Tolerance Threshold ({1 - tolerance})")
    plt.title("Class Confidence Distribution")
    plt.xlabel("Confidence Level")

    plt.legend()
    plt.savefig(save_path)
    plt.show()

# 6. Розрахунок та побудова критерію Кульбака для дельти та радіуса
def calculate_kullback_criterion(predictions, labels, delta):
    classes = np.unique(labels)
    kullback_values = []
    
    for cls in classes:
        class_preds = predictions[labels == cls]
        within_tolerance = np.sum((class_preds >= (1 - delta)))
        kullback_value = within_tolerance / len(class_preds)
        kullback_values.append(kullback_value)
    
    return np.mean(kullback_values)

def plot_kullback_vs_delta(classifier, X, y, delta_range=np.linspace(0.1, 0.5, 10), save_path="kullback_vs_delta.png"):
    kullback_scores = []
    for delta in delta_range:
        predictions = classifier.predict_proba(X).max(axis=1)
        kullback_score = calculate_kullback_criterion(predictions, y, delta)
        kullback_scores.append(kullback_score)
    
    plt.plot(delta_range, kullback_scores, marker='o')
    plt.xlabel("Delta")
    plt.ylabel("Критерій Кульбака")
    plt.title("Залежність критерію Кульбака від параметра Delta")
    plt.savefig(save_path)
    plt.show()

def plot_kullback_vs_radius(X, y, radius_range=range(10, 110, 10), save_path="kullback_vs_radius.png"):
    kullback_scores = []
    for radius in radius_range:
        within_radius = []
        for label in np.unique(y):
            class_data = X[y == label]
            center = class_data.mean(axis=0)
            distances = np.linalg.norm(class_data - center, axis=1)
            within_radius.append(np.sum(distances <= radius) / len(class_data))
        
        kullback_score = -np.sum([p * np.log(p) for p in within_radius if p > 0])
        kullback_scores.append(kullback_score)
    
    plt.plot(radius_range, kullback_scores, marker='o')
    plt.xlabel("Радіус контейнерів")
    plt.ylabel("Критерій Кульбака")
    plt.title("Залежність критерію Кульбака від радіуса контейнерів")
    plt.savefig(save_path)
    plt.show()

# 7. Класифікація за допомогою декурсивного бінарного дерева
def recursive_binary_tree_classification(X, y, max_depth=3):
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree_classifier.fit(X, y)
    return tree_classifier

# Основний запуск
image_name = "01"
image_path = f'input_images/{image_name}.jpeg'
image = cv2.imread(image_path)
frame_size = 30
tolerance = 0.2

classifier = train_classifier(X, y, n_estimators=100, max_depth=5, learning_rate=0.1)
terrain_prediction = predict_terrain_with_tolerance(classifier, image, frame_size, tolerance)
overlay_predictions(image, terrain_prediction, frame_size, f"output_images/{image_name}.jpeg")

# Побудова та збереження розподілу впевненості класів
plot_class_confidences(classifier.predict_proba(X), tolerance, "class_confidence_distribution.png")

# Побудова графіків критерію Кульбака
delta_range = np.linspace(0.1, 0.5, 10)
plot_kullback_vs_delta(classifier, X, y, delta_range, "kullback_vs_delta.png")

radius_range = range(10, 110, 10)
plot_kullback_vs_radius(X, y, radius_range, "kullback_vs_radius.png")

# Побудова декурсивного бінарного дерева
tree_classifier = recursive_binary_tree_classification(X, y, max_depth=3)
