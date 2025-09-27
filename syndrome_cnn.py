import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW # Using Adam with Weight Decay

def build_sequential_syndrome_cnn(syndrome_length: int, n: int, learning_rate: float = 1e-4) -> models.Sequential:
    """
    Builds an optimized 1D CNN architecture for decoding using the Sequential API.
    
    This model is designed to be deep enough to learn complex patterns in the
    syndrome while being regularized to prevent overfitting.
    
    Args:
        syndrome_length (int): The length of the syndrome vector (n - k).
        n (int): The length of the codeword, which is the size of the output vector.
        learning_rate (float): The learning rate for the optimizer.
        
    Returns:
        tf.keras.models.Sequential: The compiled, untrained Keras model.
    """
    
    model = models.Sequential([
        # Input Layer and Reshaping
        # The input_shape is (syndrome_length, 1) for a 1D convolution on the syndrome vector.
        layers.Input(shape=(syndrome_length,)),
        layers.Reshape((syndrome_length, 1)),
        
        # --- Convolutional Block 1 ---
        # Goal: Detect low-level local patterns in the syndrome.
        layers.Conv1D(filters=64, kernel_size=5, padding='same'),
        layers.BatchNormalization(), # Stabilizes learning
        layers.LeakyReLU(alpha=0.01), # Prevents "dying ReLU" problem
        layers.MaxPooling1D(pool_size=2), # Downsamples
        
        # --- Convolutional Block 2 ---
        # Goal: Learn more complex patterns from the features of the first block.
        layers.Conv1D(filters=128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.3), # Regularization
        
        # --- Convolutional Block 3 ---
        # Goal: Final feature extraction at a higher level of abstraction.
        layers.Conv1D(filters=256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        
        # --- Classifier Head ---
        # Flatten the learned features into a single vector for the dense layers.
        layers.Flatten(),
        
        # --- Dense Block ---
        # Goal: Translate the learned features into a prediction.
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.5), # Higher dropout before the final layer
        
        # --- Output Layer ---
        # Produces the final error vector prediction with 'n' neurons.
        # 'sigmoid' outputs a probability (0 to 1) for each bit independently.
        layers.Dense(n, activation='sigmoid')
    ], name='SyndromeDecoderCNN_Sequential')
    
    # --- Compile the Model ---
    # AdamW is often superior to Adam as it decouples weight decay from optimization.
    optimizer = AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['binary_accuracy'])
                  
    return model

# --- Example Usage ---
if __name__ == '__main__':
    # Using the real-world parameters for demonstration
    N = 3488
    K = 2720
    SYNDROME_LENGTH = N - K # 768

    # Build the model
    model = build_sequential_syndrome_cnn(syndrome_length=SYNDROME_LENGTH, n=N)
    
    # Print the model summary to see the architecture
    model.summary()
