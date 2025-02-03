from keras import Model
from keras.src.layers import GlobalAveragePooling2D, Dropout, Dense
from model_selection import resnet50, densenet121, mobilenetv3


# Function to modify pre-trained models
def modify_model(base_model, num_classes):
    # Add a global average pooling layer
    x = GlobalAveragePooling2D()(base_model.output)

    # Add a dropout layer to prevent overfitting
    x = Dropout(0.5)(x)

    # Add a dense classification layer
    output = Dense(num_classes, activation='softmax')(x)

    # Combine the base model with the new layers
    model = Model(inputs=base_model.input, outputs=output)
    return model


# Modify models for 3 classes
resnet_model = modify_model(resnet50, num_classes=3)
densenet_model = modify_model(densenet121, num_classes=3)
mobilenet_model = modify_model(mobilenetv3, num_classes=3)

print("Models modified successfully!")
