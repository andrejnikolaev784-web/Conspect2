import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = {
    'text': [
        # Класс 0 - Финтех / кредиты
        "Банковский кредитный конвейер оценивает риски дефолта заемщика.",
        "Алгоритм скоринга рассчитывает вероятность невозврата кредита.",
        "Финансовый аналитик оценивает кредитоспособность клиента.",
        "Банк анализирует доход клиента перед выдачей займа.",
        "Модель прогнозирует риск просрочки по кредиту.",

        # Класс 1 - Геймдев / игровой ИИ
        "Игровой агент обучается методу Q-learning в трехмерной симуляции.",
        "Разработчик настраивает алгоритм Minimax для шахматного бота.",
        "Нейросеть получает награды и штрафы от игровой среды.",
        "Игровой персонаж выбирает оптимальное действие в лабиринте.",
        "Искусственный интеллект управляет противником в компьютерной игре.",

        # Класс 2 - Медицина
        "Врач анализирует результаты медицинского обследования пациента.",
        "Нейросеть помогает обнаруживать признаки заболевания на снимке.",
        "Модель прогнозирует вероятность развития болезни.",
        "Медицинская система анализирует показатели крови пациента.",
        "Алгоритм используется для диагностики заболеваний.",

        # Класс 3 - Образование
        "Студент проходит онлайн-курс по программированию.",
        "Преподаватель проверяет результаты контрольной работы.",
        "Образовательная платформа рекомендует студенту новые задания.",
        "Нейросеть анализирует успеваемость учащихся.",
        "Система помогает подобрать учебный материал для студента.",

        # Класс 4 - Киберспорт
        "Киберспортсмен тренируется перед важным турниром.",
        "Команда анализирует статистику матчей соперника.",
        "Игрок улучшает реакцию и точность стрельбы.",
        "Тренер изучает стратегию команды противника.",
        "Система анализирует действия игроков во время матча."
    ],
    'label': [
        # Финтех
        0, 0, 0, 0, 0,

        # Геймдев
        1, 1, 1, 1, 1,
        # Медицина
        2, 2, 2, 2, 2,
        # Образование
        3, 3, 3, 3, 3,
        # Киберспорт
        4, 4, 4, 4, 4
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['text']).toarray()
y = np.array(df['label'])

print("Размерность матрицы признаков:", X.shape)
print("Количество классов:", len(np.unique(y)))

X_train = X[:20]
y_train = y[:20]

X_test = X[20:]
y_test = y[20:]

model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dense(16, activation='relu'),
    # 5 нейронов - по одному на каждый класс
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    verbose=0
)

train_loss, train_acc = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nТочность на обучающих данных: {train_acc:.2%}")
print(f"Точность на тестовых данных: {test_acc:.2%}")

class_names = [
    "Финтех / Кредиты",
    "Геймдев / Игровой ИИ",
    "Медицина",
    "Образование",
    "Киберспорт"
]

new_sentences = [
    "Банк рассчитывает вероятность выдачи кредита клиенту.",
    "Игровой бот выбирает следующий ход в шахматной партии.",
    "Врач использует нейросеть для диагностики пациента.",
    "Студент выполняет задания на образовательной платформе.",
    "Команда готовится к финальному матчу киберспортивного турнира."
]

new_vectors = vectorizer.transform(new_sentences).toarray()

predictions = model.predict(new_vectors, verbose=0)

for sentence, prediction in zip(new_sentences, predictions):

    predicted_class = np.argmax(prediction)
    probability = prediction[predicted_class]

    print("\nПредложение:", sentence)
    print("Класс:", class_names[predicted_class])
    print("Вероятность:", f"{probability:.2%}")