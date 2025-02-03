import pickle
from keras.src.applications.resnet import ResNet50
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

# Train ResNet50
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model = modify_base_model(resnet_base, num_classes=3, freeze_layers=100)
resnet_history = compile_and_train_model(
    resnet_model,
    train_dataset,
    val_dataset,
    save_path=f'{root}/Models/resnet50_best_checkpoint.keras',
    epochs=50
)

# Save the model
resnet_model.save(f'{root}/Models/resnet50_model.keras')
print("ResNet50 model trained and saved successfully!")

# Save training history
history_path = f'{root}/Models/resnet50_training_history.pkl'
save_training_history(resnet_history, history_path)
