import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Parameters for ViT
img_size = 28
num_classes = 10
patch_size = 7  # 28/7=4 patches per side, 16 patches total
num_patches = (img_size // patch_size) ** 2
embedding_dim = 64
num_heads = 4
transformer_layers = 4
mlp_dim = 128

# Helper: create patches
def extract_patches(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=tf.expand_dims(images, -1),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [batch_size, -1, patch_size*patch_size])
    return patches

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Model: Vision Transformer for MNIST
inputs = tf.keras.Input(shape=(img_size, img_size))
patches = tf.keras.layers.Lambda(lambda x: extract_patches(x, patch_size))(inputs)
patch_embeddings = tf.keras.layers.Dense(embedding_dim)(patches)

# Add positional encoding
positions = tf.range(start=0, limit=num_patches, delta=1)
positional_encoding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)
encoded_patches = patch_embeddings + positional_encoding

x = encoded_patches
for _ in range(transformer_layers):
    # Multi-head self-attention
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    # MLP block
    mlp_output = tf.keras.layers.Dense(mlp_dim, activation='relu')(x)
    mlp_output = tf.keras.layers.Dense(embedding_dim)(mlp_output)
    x = tf.keras.layers.Add()([x, mlp_output])
    x = tf.keras.layers.LayerNormalization()(x)

# Classification head
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(mlp_dim, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and checkpoint
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_mnist_transformer.h5', save_best_only=True)
]

# Train model
history = model.fit(x_train, y_train_cat, epochs=20, batch_size=64, validation_split=0.1, callbacks=callbacks)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate model
score = model.evaluate(x_test, y_test_cat)
print(f"Test accuracy: {score[1]:.4f}")
print(f"Test loss: {score[0]:.4f}")

# Predict and report
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save final model
model.save('final_mnist_transformer.h5')
print("Model saved as final_mnist_transformer.h5")

