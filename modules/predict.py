import os
import pandas as pd
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def load_model(model_path: Path):
    logging.info(f"Loading model from {model_path}")
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info(f"Loaded model type: {type(model)}")
    logging.info("Model loaded successfully")
    return model


def load_test_data(file_path: Path):
    logging.info(f"Loading data from {file_path}")
    try:
        data = pd.read_json(file_path, typ='series')
        logging.info(f"Data loaded successfully with type {type(data)}")
    except ValueError as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
        raise

    if isinstance(data, pd.Series):
        data = pd.DataFrame([data])
    elif isinstance(data, dict):
        data = pd.DataFrame([data])

    logging.info(f"Data type after conversion: {type(data)}")
    logging.info(f"Data head: {data.head() if isinstance(data, pd.DataFrame) else data}")

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Data in {file_path} is not a valid format for DataFrame")

    return data


def generate_features(data: pd.DataFrame):
    # Пример генерации признаков
    data['feature1'] = data['odometer']
    data['feature2'] = data['year']
    # Добавьте больше логики генерации признаков, если необходимо
    return data[['feature1', 'feature2']]


def make_predictions(model, test_data_path: Path):
    predictions = []
    logging.info(f"Making predictions for files in {test_data_path}")
    for file_name in os.listdir(test_data_path):
        file_path = test_data_path / file_name
        logging.info(f"Processing file: {file_path}")
        data = load_test_data(file_path)
        data_prepared = generate_features(data)
        pred = model.predict(data_prepared)
        pred_df = pd.DataFrame(pred, columns=['prediction'])
        pred_df['file'] = file_name
        predictions.append(pred_df)
    return pd.concat(predictions)


def save_predictions(predictions, output_path: Path):
    output_file = output_path / 'predictions.csv'
    logging.info(f"Saving predictions to {output_file}")
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    logging.info("Predictions saved successfully")


def predict():
    model_path = Path(r'C:\Users\America\airflow_hw\models\best_model.pkl')
    test_data_path = Path(r'C:\Users\America\airflow_hw\data\test')
    output_path = Path(r'C:\Users\America\airflow_hw\data\predictions')

    model = load_model(model_path)
    predictions = make_predictions(model, test_data_path)
    save_predictions(predictions, output_path)


if __name__ == '__main__':
    predict()
