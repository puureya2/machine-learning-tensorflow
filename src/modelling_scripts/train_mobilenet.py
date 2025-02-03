import pickle
from keras.src.applications.mobilenet_v3 import MobileNetV3Large
from keras.src.utils import image_dataset_from_directory
from model_training import modify_base_model, compile_and_train_model
from src.utils import root_dir as root


# Function to save training history using Pickle
def save_training_history(history, file_path):
    """
    Saves the training history using Pickle.

    :param history: History object from model.fit()
    :param file_path: Path to save the Pickle file
    """
    history_dict = history.history
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(history_dict, pickle_file)
    print(f"Training history saved to {file_path}")


# Load datasets
train_dataset = image_dataset_from_directory(
    f'{root}/Data/split/train',
    image_size=(224, 224),
    batch_size=64
)

val_dataset = image_dataset_from_directory(
    f'{root}/Data/split/validate',
    image_size=(224, 224),
    batch_size=64
)

# Train MobileNetV3
mobilenet_base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_model = modify_base_model(mobilenet_base, num_classes=3, freeze_layers=100)
mobilenet_history = compile_and_train_model(
    mobilenet_model,
    train_dataset,
    val_dataset,
    save_path=f'{root}/Models/mobilenetv3_best_checkpoint.keras',
    epochs=50
)

mobilenet_model.save(f'{root}/Models/mobilenetv3_model.keras')
print("MobileNetV3 model trained and saved successfully!")

# Save training history
save_training_history(mobilenet_history, f'{root}/Models/mobilenetv3_training_history.pkl')
