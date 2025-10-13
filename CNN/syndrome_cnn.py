import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow_addons.optimizers import AdamW

def build_sequential_syndrome_cnn(syndrome_length: int, n: int, learning_rate: float = 5e-5) -> Sequential:
    """
    Builds a more lightweight 1D CNN architecture to reduce memory usage and
    potentially improve training on complex, sparse problems.
    
    Args:
        syndrome_length (int): The length of the binarized syndrome vector.
        n (int): The length of the codeword (output vector size).
        learning_rate (float): The learning rate for the optimizer.
    
    Returns:
        tf.keras.models.Sequential: The compiled, untrained Keras model.
    """
    
    model = Sequential([
        # --- Feature Extraction Backbone ---
        Reshape((syndrome_length, 1), input_shape=(syndrome_length,)),
        
        # --- Convolutional Block 1 ---
        Conv1D(filters=64, kernel_size=5, padding='same', name='conv1'),
        BatchNormalization(name='bn1'),
        LeakyReLU(alpha=0.01, name='lrelu1'),
        MaxPooling1D(pool_size=2, name='pool1'),
        
        # --- Convolutional Block 2 (Simplified) ---
        # The third convolutional block has been removed to simplify the model.
        Conv1D(filters=128, kernel_size=3, padding='same', name='conv2'),
        BatchNormalization(name='bn2'),
        LeakyReLU(alpha=0.01, name='lrelu2'),
        Dropout(0.4, name='dropout2'),
        
        # --- Classifier Head ---
        Flatten(name='flatten'),
        
        # --- Dense Block (Simplified) ---
        # Reduced the number of neurons to lower parameter count and memory footprint.
        Dense(256, name='dense1'),
        BatchNormalization(name='bn_dense1'),
        LeakyReLU(alpha=0.01, name='lrelu_dense1'),
        Dropout(0.6, name='dropout_dense1'),
        
        # --- Output Layer ---
        Dense(n, activation='sigmoid', name='output_error_vector')
    ], name='SyndromeDecoderCNN_Sequential_Light') # Renamed for clarity
    
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-7)
    
    model.compile(optimizer=optimizer, 
                  loss=tfa.losses.SigmoidFocalCrossEntropy(), 
                  metrics=['binary_accuracy', 'mse'])
                  
    return model

