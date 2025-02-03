import pickle
from keras.src.applications.densenet import DenseNet121
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

# Train DenseNet121
densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
densenet_model = modify_base_model(densenet_base, num_classes=3, freeze_layers=100)
densenet_history = compile_and_train_model(
    densenet_model,
    train_dataset,
    val_dataset,
    save_path=f'{root}/Models/densenet121_best_checkpoint.keras',
    epochs=50
)

densenet_model.save(f'{root}/Models/densenet121_model.keras')
print("DenseNet121 model trained and saved successfully!")

# Save training history
save_training_history(densenet_history, f'{root}/Models/densenet121_training_history.pkl')
