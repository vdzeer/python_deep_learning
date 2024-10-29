import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

def label_image_quick(image, frame_size):
    h, w, _ = image.shape
    grid_h, grid_w = h // frame_size, w // frame_size
    labels = np.zeros((grid_h, grid_w), dtype=int)
    
    # Індекс кадру для маркування
    current_frame = [0, 0]  # [i, j]

    # Функція для обробки натискань клавіш
    def on_key(event):
        nonlocal current_frame
        if event.key in ['0', '1', '2']:
            labels[current_frame[0], current_frame[1]] = int(event.key)
            plt.close()
            
            # Переходимо до наступного кадру
            if current_frame[1] < grid_w - 1:
                current_frame[1] += 1
            elif current_frame[0] < grid_h - 1:
                current_frame[0] += 1
                current_frame[1] = 0
            else:
                current_frame[0] = -1

    # Цикл для показу кожного кадру та очікування натискання клавіші
    while current_frame[0] != -1:
        i, j = current_frame
        frame = image[i*frame_size:(i+1)*frame_size, j*frame_size:(j+1)*frame_size]
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Виберіть мітку для кадру ({i}, {j}) - натисніть 0 (ліс), 1 (поле), 2 (дорога)")
        plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    return labels

def generate_dataset(image, frame_size, labels):
    h, w, _ = image.shape
    data = []

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            y, x = i * frame_size, j * frame_size
            frame = image[y:y+frame_size, x:x+frame_size]
            avg_color = frame.mean(axis=(0, 1))
            label = labels[i, j]  # клас місцевості з масиву міток

            data.append([avg_color[0], avg_color[1], avg_color[2], label])

    # Перетворюємо в DataFrame для зручного збереження
    df = pd.DataFrame(data, columns=["R", "G", "B", "Label"])
    return df

# Основна функція для додавання нового зображення до датасету
def add_image_to_dataset(image_path, frame_size, csv_file="terrain_dataset.csv"):
    # Завантажуємо зображення
    image = cv2.imread(image_path)
    labels = label_image_quick(image, frame_size)

    # Генеруємо датасет
    dataset = generate_dataset(image, frame_size, labels)

    # Додаємо нові дані до загального CSV-файлу
    try:
        # Завантажуємо існуючий датасет, якщо файл вже існує
        existing_data = pd.read_csv(csv_file)
        # Об’єднуємо існуючі дані з новими
        dataset = pd.concat([existing_data, dataset], ignore_index=True)
    except FileNotFoundError:
        # Якщо файл не існує, просто зберігаємо новий датасет
        pass

    # Зберігаємо оновлений датасет
    dataset.to_csv(csv_file, index=False)
    print(f"Додано дані з {image_path} до {csv_file}")

# Шлях до папки з зображеннями
image_folder = "test_imgs/"

# Отримуємо список всіх зображень у папці
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

print("Знайдені зображення:", image_paths)

frame_size = 50  # Розмір кадру

# Додаємо кожне зображення з папки до датасету
for image_path in image_paths:
    add_image_to_dataset(image_path, frame_size)
