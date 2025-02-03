from keras import Model
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import GlobalAveragePooling2D, Dense
from keras.src.optimizers import Adam


# Modify the base model
def modify_base_model(base_model, num_classes, freeze_layers=100):
    """
    Modify a base model by freezing layers and adding custom layers.

    Args:
        base_model: A pre-trained model to modify.
        num_classes: Number of classes for the classification task.
        freeze_layers: Number of layers to freeze in the base model.

    Returns:
        Modified model ready for training.
    """
    # Freeze specified layers
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


# Compile and train the model
def compile_and_train_model(model, train_ds, val_ds, save_path, epochs=50):
    """
    Compile and train a model with checkpoints and early stopping.

    Args:
        model: The model to train.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        save_path: Filepath to save the best model.
        epochs: Number of epochs for training.

    Returns:
        History object containing training metrics.
    """
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1
    )
    return history
