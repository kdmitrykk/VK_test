# VK_test
# Поиск коротких заставок в сериалах

Этот проект реализует полный пайплайн для автоматического определения короткой интро-заставки (1–10 секунд) в начале каждой серии.  
Задача сводится к тому, чтобы по видео- и аудиопотокам вывести относительные метки начала и конца заставки.

---

## Структура файла

Весь код собран в одном Jupyter Notebook и включает следующие разделы:

1. **Импорт и настройка**  
   - Загрузка зависимостей (`torch`, `opencv-python`, `ffmpeg-python`, `numpy`, `pandas`, `tqdm`)  
   - Распаковка архивов `train_videos.zip` и `test_videos.zip`  
   - Задание констант: `SEQ_LEN`, `FPS`, `DEVICE`, `NUM_EPOCHS`, пути к данными  

2. **Вспомогательные функции**  
   - `to_seconds(hh_mm_ss: str) → float` — перевод временных меток в секунды  
   - `smooth_predictions(preds: np.ndarray) → np.ndarray` — скользящая средняя или гауссов фильтр по вероятностям  
   - `extract_best_interval(smoothed: np.ndarray, fps: int) → (start_rel: float, end_rel: float)` — выбор непрерывного фрагмента выше порога  

3. **Класс датасета `IntroDataset`**  
   - **Инициализация**  
     - `root_dir: str` — папка с `.mp4`  
     - `labels_json: str` — JSON с разметкой `{ "episode.mp4": {"start": "00:00:05", "end": "00:00:12"} }`  
     - `seq_len: int`, `fps: int`, `train: bool`  
   - **Метод `__getitem__`**  
     - Чтение кадров через OpenCV, ресайз до нужного разрешения  
     - Извлечение аудиофичей (усреднённая амплитуда по `seq_len` сегментам)  
     - Нормализация и упаковка в тензоры  
     - В режиме `train` возвращает кортеж `(video_feats, audio_feats, rel_times, video_id)`, в режиме `test` — `(video_feats, audio_feats, video_id)`  

4. **Collate-функция `custom_collate`**  
   - Принимает батч из кортежей разной длины по временной оси  
   - Паддинг всех видеопоследовательностей до `T_max` (максимальной длины в батче)  
   - Паддинг аудиофичей аналогично  
   - Склеивает `video_feats: Tensor[B×T_max×C×H×W]`, `audio_feats: Tensor[B×T_max×1]`, `rels: Tensor[B×2]` и список `video_ids`  

5. **Модель `IntroDetector`**  
   - **Архитектура**  
     1. Бэкбон `ResNet18` (предобученный) для извлечения признаков из каждого кадра  
     2. Конкатенация видеопризнаков (размер `video_dim`) и аудиопризнаков (`audio_dim=1`)  
     3. Двунаправленный LSTM (`hidden_size=128`, 2 слоя)  
     4. Два “выхода”:  
        - **Регрессия** относительных меток `[start_rel, end_rel]` (последний скрытый → Linear → ReLU → Linear)  
        - **Классификация** наличия интро (последний скрытый → Linear → Sigmoid → `presence_prob`)  

6. **Обучение**  
   1. Создать `train_loader` с `batch_size=4`, `shuffle=True`, `collate_fn=custom_collate`.  
   2. Инициализировать модель на `DEVICE` и определить оптимизатор `Adam(lr=1e-4)`.  
   3. Задать лосс-функции:  
      - `criterion_reg = MSELoss()` для регрессии  
      - `criterion_cls = BCELoss()` для классификации  
   4. Для каждой эпохи (`1…NUM_EPOCHS`):  
      - Переключить модель в режим `train()`  
      - Для каждого батча рассчитать `pred_time, pred_presence = model(video, audio)`  
      - Суммарный лосс = `0.7*loss_reg(pred_time, true_time) + 0.3*loss_cls(pred_presence, true_label)`  
      - `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`  
   5. В конце эпохи провести `eval_epoch` на валидации и сохранить чекпоинт с наименьшим `val_loss`  

7. **Валидация**  
   - Аналогично обучению, без градиентов (`with torch.no_grad()`)  
   - Считаются:  
     - Средняя абсолютная ошибка для `start` и `end`  
     - Точность классификации интро  

8. **Early stopping и сохранение**  
   - Если `val_loss` ниже `best_loss` → сохранить `best_model.pth`  
   - Если после 10 эпох нет улучшений (val_loss > best_loss × 1.1) → прервать обучение  

9. **Получение результатов**  
   1. Загрузить `IntroDataset(test, train=False)` и `best_model.pth`  
   2. Для каждого видео вывести `pred_time, pred_presence`  
   3. Сгладить `pred_time` через `smooth_predictions`  
   4. Преобразовать относительные метки в секунды и собрать словарь `{video_id: [start_sec, end_sec]}`  
   5. Сохранить в `test_predictions.json`  

---

## Как запустить

1. Открыть Notebook и выполнить ячейку с распаковкой данных (`train_videos.zip` и `test_videos.zip`) в папку `data/`.  
2. Запустить ячейку “Обучение” и дождаться окончания.  
3. После завершения обучения выполнить ячейку “Инференс” — результат будет в `data/test_predictions.json`.  

---

## Зависимости

```text
python
torch
torchvision
numpy
pandas
opencv-python
ffmpeg-python
tqdm
