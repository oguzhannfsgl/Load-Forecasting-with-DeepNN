import tensorflow as tf

# Defining network and its layers b
class CustomModel(tf.keras.models.Model):
    def __init__(self, embedding_dim=8, vector_len=32, hidden_units=16, bi_dir=False, output_n=24):
        super().__init__()
        self.vector_len = vector_len
        self.hidden_units = hidden_units
        self.bi_dir = bi_dir
    
        # Defining a random load vector
        self.load_vector = self.add_weight(shape=(vector_len,), initializer="random_normal", trainable=True)
        
        # Defining embedding layers for categorical features
        self.dow_embedding = tf.keras.layers.Embedding(input_dim=7, output_dim=embedding_dim)
        self.dom_embedding = tf.keras.layers.Embedding(input_dim=32, output_dim=embedding_dim)
        self.hour_embedding = tf.keras.layers.Embedding(input_dim=24, output_dim=embedding_dim)
        self.is_holiday_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=embedding_dim)
        
        # FCN for feature fusion
        self.feature_fc = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="relu"),
                                              tf.keras.layers.Dense(16, activation="relu")])
        # LSTM
        self.lstm = tf.keras.layers.LSTM(units=hidden_units, return_sequences=False)
        if bi_dir:
            self.lstm = tf.keras.layers.Bidirectional(self.lstm)
        
        # Last two FC layers
        self.fc1 = tf.keras.layers.Dense(16, activation="relu")
        self.fc_final = tf.keras.layers.Dense(units=output_n, activation="sigmoid")
        
    def call(self, inputs):
        # Forward propagation of the network
        feature_vector_load = tf.expand_dims(inputs[:,:,0], axis=-1) * self.load_vector
        feature_vector_dow = self.dow_embedding(inputs[:,:,1])
        feature_vector_hour = self.hour_embedding(inputs[:,:,3])
        feature_vector_dom = self.dom_embedding(inputs[:,:,4])
        feature_vector_is_holiday = self.is_holiday_embedding(inputs[:,:,5])
        
        feature_stack = [feature_vector_load, feature_vector_dow, feature_vector_dom, feature_vector_hour,
                         feature_vector_is_holiday]
        feature_vectors = tf.concat(feature_stack, axis=-1)

        feature_vectors = self.feature_fc(feature_vectors)
        out_lstm = self.lstm(feature_vectors)
        out_att = out_lstm
        outputs = self.fc_final(self.fc1(out_att))
        
        return outputs
    
    
    
class DeepEnergy(tf.keras.models.Model):
    """
    A High Precision Artificial Neural Networks Model for Short-Term Energy Load Forecasting.
        Ping-Huan Kuo and Chiou-Jye Huang
    """
    
    def __init__(self, input_n=24, output_n=24):
        super().__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.conv_block = tf.keras.Sequential([tf.keras.layers.Conv1D(16, kernel_size=9, padding="same", activation="relu"),
                                              tf.keras.layers.MaxPooling1D(2),
                                              tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
                                              tf.keras.layers.MaxPooling1D(2),
                                              tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
                                              tf.keras.layers.MaxPooling1D(2)])
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.Sequential([tf.keras.layers.Dropout(0.1),
                                       tf.keras.layers.Dense(output_n, activation="sigmoid")])
        
    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.input_n, 8))
        out_conv = self.conv_block(inputs)
        out_flatted = self.flatten(out_conv)
        out = self.fc(out_flatted)
        return out
    
    

class SeqMlp(tf.keras.models.Model):
    """
    A Two-Stage Short-Term Load Forecasting Method Using Long Short-Term Memory and Multilayer Perceptron
    
    seq features: "total_load_previous", "month_of_year", "day_of_week", "hour", "total_load_1_week"
    """
    def __init__(self, input_n=24):
        super().__init__()
        self.input_n = input_n
        
        self.lstm = tf.keras.layers.LSTM(units=100, return_sequences=True)#units=100
        self.att = tf.keras.layers.Attention()
        self.seq_fc = tf.keras.Sequential([tf.keras.layers.Dense(64, activation="relu"),#units=64
                                            tf.keras.layers.Dense(1, activation="sigmoid")])
        
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(10, activation="relu"),
                                       tf.keras.layers.Dense(100, activation="relu"),
                                       tf.keras.layers.Dense(100, activation="relu"),
                                       tf.keras.layers.Dense(1, activation="sigmoid")])
        
    def call(self, inputs, training=True):
        input_seq, input_mlp = inputs[:,:,:5], inputs[:,:,2:]
        out_seq = self.lstm(input_seq)
        out_seq = self.att([out_seq, out_seq, out_seq])
        out_seq = self.seq_fc(out_seq)# output shape (bs, seq, 1)
        input_mlp = tf.concat([input_mlp, out_seq], axis=-1)# shape (bs, seq, x)
        input_mlp = tf.reshape(input_mlp, (-1, 7))
        out_mlp = self.mlp(input_mlp)
        out_mlp = tf.reshape(out_mlp, (-1, self.input_n))
        return out_mlp