import pandas as pd
import joblib
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
from typing import List
import os
from datetime import datetime
import argparse
import uvicorn

# Инициализация приложения FastAPI
app = FastAPI(
    title="API для предсказания риска сердечного приступа",
    description="Загрузите CSV файл с данными пациентов для получения предсказаний риска сердечного приступа (вероятность от 0 до 1)."
)

# Загрузка модели
try:
    model_filename = 'rfc.joblib'
    model = joblib.load(model_filename)
    
    # Получаем признаки из модели
    if hasattr(model, 'feature_name_'):
        model_features = model.feature_name_
    else:
        model_features = [
            'Age', 'Cholesterol', 'Heart_rate', 'Diabetes', 'Family_History',
            'Obesity', 'Alcohol_Consumption', 'Exercise_Hours_Per_Week',
            'Diet', 'Medication_Use', 'Stress_Level', 'Sedentary_Hours_Per_Day',
            'BMI', 'Triglycerides', 'Physical_Activity_Days_Per_Week',
            'Sleep_Hours_Per_Day', 'Blood_sugar', 'Systolic_blood_pressure',
            'Diastolic_blood_pressure'
        ]

except FileNotFoundError:
    raise RuntimeError(f"Файл модели '{model_filename}' не найден.")
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}")

# HTML-форма для загрузки
@app.get("/", response_class=HTMLResponse)
async def serve_upload_form():
    return """
     <!DOCTYPE html>
     <html lang="ru">
     <head>
     <meta charset="UTF-8">
     <title>Приложение для предсказания риска сердечного приступа</title>
     <style>
     body {
     margin: 0;
     padding: 0;
     width: 100%;
     height: 100%;
     background-color: white; /* Белый фон страницы */
     display: flex;
     justify-content: center;
     align-items: center;
     flex-direction: column;
     }
    .container {
     width: 90%;
     max-width: 800px;
     padding: 40px;
     text-align: center;
     border: 2px solid #eeaed0; /* Рамка цвета фона страницы */
     background-color: #fcebf3; /* Цвет фона внутри рамки */
     border-radius: 8px;
     }
     h1 {
     margin-bottom: 20px;
     font-family: Arial, sans-serif;
     color: #430E2A;
     }
     p {
     margin-bottom: 20px;
     font-size: 18px;
     color: #430E2A; /* Черный цвет текста */
     }
     form {
     display: flex;
     flex-direction: column;
     align-items: center;
     }
     input[type="file"] {
     margin-bottom: 20px;
     }
     input[type="submit"] {
     padding: 10px 20px;
     font-size: 16px;
     background-color: white;
     border: 1px solid #eeaed0;
     border-radius: 4px;
     cursor: pointer;
     }
     input[type="submit"]:hover {
     background-color: #f3c3db;
     }
     </style>
     </head>
     <body>
     <div class="container">
     <h1>Приложение для предсказания риска сердечного приступа</h1>
     <p>Для предсказания необходимо загрузить файл в формате 'csv'. После загрузки файла нажмите кнопку "Предсказать".</p>
     <form action="/predict" enctype="multipart/form-data" method="post">
     <input name="file" type="file" accept=".csv">
     <input type="submit" value="Предсказать">
     </form>
     </div>
     </body>
     </html>
 """

# Эндпоинт для предсказания и сохранения в CSV
@app.post("/predict")
async def predict_and_save(file: UploadFile = File(...)):
    try:
        # Читаем файл
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        data_1 = pd.read_csv(buffer)
        
        # Обработка данных
        rename_dict = {
            'Family History': 'Family_History',
            'Alcohol Consumption': 'Alcohol_Consumption',
            'Exercise Hours Per Week': 'Exercise_Hours_Per_Week',
            'Medication Use': 'Medication_Use',
            'Stress Level': 'Stress_Level',
            'Sedentary Hours Per Day': 'Sedentary_Hours_Per_Day',
            'Physical Activity Days Per Week': 'Physical_Activity_Days_Per_Week',
            'Sleep Hours Per Day': 'Sleep_Hours_Per_Day',
            'Blood sugar': 'Blood_sugar',
            'Systolic blood pressure': 'Systolic_blood_pressure',
            'Diastolic blood pressure': 'Diastolic_blood_pressure',
            'Heart rate': 'Heart_rate'
        }
        
        data = data_1.rename(columns=rename_dict)
        
        # Проверяем наличие всех необходимых столбцов
        missing_cols = set(model_features) - set(data.columns)
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Отсутствуют столбцы: {', '.join(missing_cols)}"
            )
        
        # Фильтруем и упорядочиваем данные
        data = data[model_features]
                
        # Делаем предсказания
        predictions = model.predict_proba(data)[:, 1]
        predictions_df = pd.DataFrame(predictions, columns = ['predictions'])

        #Устанавливаем порог классификации
        predictions_bi = predictions_df['predictions'].apply(lambda x: '1'  if x >= 0.79 else '0')

        # Создаем датафрейм с результатами
        results_df = pd.DataFrame({'Record_Number': range(1, len(predictions_bi) + 1), 'prediction': predictions_bi})
        
        # Добавляем предсказания к исходным данным
        final_df = pd.concat([data_1, results_df], axis=1)

        #Выделяем два столбца для конечного результата
        final_df = final_df[['id', 'prediction']]
        
        
        # Формируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.csv"
        filepath = os.path.join("results", filename)
        
        # Создаем директорию для результатов, если её нет
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Сохраняем результаты в CSV
        final_df.to_csv(filepath, index=False)
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Обработка ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return HTMLResponse(
        content=f"""
        <h1>Ошибка:</h1>
        <p>{exc.detail}</p>
        <a href="/">Вернуться</a>
        """,
        status_code=exc.status_code
    )

# Запуск сервера
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)