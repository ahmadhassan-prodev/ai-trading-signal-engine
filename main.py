from src.data.data_fetch import fetch_data
from src.features.feature_engineering import create_features
from src.models.create_labels import create_label
from src.models.scaling import scale_features
from src.models.train_rf import train_model

def main():
    print("Running AI Trading Signal Engine...")

    fetch_data()
    create_features()
    create_label()
    scale_features()
    train_model()

if __name__ == "__main__":
    main()