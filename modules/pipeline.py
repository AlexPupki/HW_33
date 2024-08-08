import os
import pandas as pd
import logging
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression  # Пример модели, замените на вашу

logging.basicConfig(level=logging.INFO)


def load_data(file_path: Path):
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Checking file: {file_path}")
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    logging.info(f"File found: {file_path}")
    return pd.read_csv(file_path)


def train_model(df):
    logging.info("Training model")
    X = df[['odometer', 'year']]  # Замените на ваши реальные признаки
    y = df['price']  # Замените 'price' на вашу целевую переменную
    model = LinearRegression()
    model.fit(X, y)
    return model


def save_model(model, file_path: Path):
    logging.info(f"Saving model to {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model saved successfully")


def pipeline():
    logging.info("Starting pipeline")

    base_dir = Path(__file__).resolve().parent.parent
    train_data_path = base_dir / 'data' / 'train' / 'train_data.csv'
    model_save_path = base_dir / 'models' / 'best_model.pkl'

    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Train data path: {train_data_path}")

    if train_data_path.exists():
        logging.info(f"File exists: {train_data_path}")
    else:
        logging.error(f"File does not exist: {train_data_path}")

    df = load_data(train_data_path)
    model = train_model(df)
    save_model(model, model_save_path)

    logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    pipeline()
