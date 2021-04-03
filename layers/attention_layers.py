import tensorflow as tf
from keras.layers import Layer
from keras import backend as K
from definitions.hyper_parameters import WINDOW_WIDTH
from keras.layers import Dense, Activation, Concatenate


##### This is Bahdanau Attention #####
class Attention_add(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention_add, self).__init__()

    def build(self, inputs):
        self.output_dimensions = inputs[2]
        self.W1 = tf.keras.layers.Dense(self.output_dimensions)
        self.W1.build(input_shape=(None, None, self.output_dimensions))  # (B, 1, H)
        self.W2 = tf.keras.layers.Dense(self.output_dimensions)
        self.W2.build(input_shape=(None, None, self.output_dimensions))  # (B, 1, H)
        self.V = tf.keras.layers.Dense(1)
        self.V.build(input_shape=(None, None, self.output_dimensions))  # (B, 1, H)

    def call(self, inputs):
        hidden_with_time_axis = inputs[:, -1]  # (B , H)
        hidden_with_time_axis = tf.expand_dims(input=hidden_with_time_axis, axis=1)  # (B, 1, H)

        # A = W_1(h_t) + W_2 (h_s)
        # V(tanhh(A))
        score = self.V(K.tanh(self.W1(inputs) + self.W2(hidden_with_time_axis)))  # (B, S, 1)
        attention_weights = K.softmax(score, axis=1)  # (B, S, 1)

        # C_t= ∑(α_ts * h_s)
        context_embedding = attention_weights * inputs  # (B, S, H)
        context_vector = tf.reduce_sum(context_embedding, axis=1)
        return context_vector


##### Uses Local Attention with a Normal Distribution Around the Window ######
class Attention_local(Layer):
    def __init__(self):  # This layer is just to start defining the layer
        super(Attention_local, self).__init__()

    def build(self, input_shape):  # This layer defines the shape of the weights and bias
        self.sequence_length = input_shape[0][1]  # the number of words (max_len)
        self.output_dimensions = input_shape[0][2]  # output dim [hidden vec dimensions]
        self.W_p = Dense(self.output_dimensions)
        self.W_p.build(input_shape=(None, None, self.output_dimensions))  # (B, 1, H)

        self.W_a = Dense(self.output_dimensions)
        self.W_a.build(input_shape=(None, None, self.output_dimensions))  # (B, 1, H)

        self.V_p = tf.keras.layers.Dense(1)
        self.V_p.build(input_shape=(None, None, self.output_dimensions))
        self.window_width = WINDOW_WIDTH
        super(Attention_local, self).build(input_shape)

    def call(self, inputs):  # This is where the action happens
        # inputs is the input tensor
        target_hidden_state = inputs[1]  # (B , H)
        source_hidden_state = inputs[0]  # (B, S, H)
        hidden_with_time_axis = tf.expand_dims(input=target_hidden_state, axis=1)  # (B, 1, H)

        # N = W_1(h_t)
        # M = V(tanh(N))
        aligned_position = self.V_p(K.tanh(self.W_p(hidden_with_time_axis)))  # (B, 1, 1)
        # p_t = sigmoid(M) * S
        aligned_position = K.sigmoid(aligned_position)  # (B, 1, 1)
        aligned_position = aligned_position * self.sequence_length  # (B, 1, 1)

        # α_t=softmax(h_t W_2 h_s)
        attention_score = K.softmax(
            source_hidden_state * self.W_a(hidden_with_time_axis))  # (B, S, H)
        attention_weights = Activation('softmax')(attention_score)  # (B, S, H)

        # α_t (s)= α_t*exp(-((s-p_t)^2)/(2σ^2)
        gaussian_estimation = lambda i: tf.exp(-2 * tf.square((i - aligned_position) / self.window_width))
        gaussian_factor = gaussian_estimation(0)
        for x in range(1, self.sequence_length):
            gaussian_factor = Concatenate(axis=1)([gaussian_factor, gaussian_estimation(x)])
        attention_weights = attention_weights * gaussian_factor  # (B, S, H)

        # C_t= ∑ (α_t (s) * h_s)
        context_embedding = attention_weights * source_hidden_state  # (B, S, H)
        # Derive context vector by getting the weighted average over the source states
        context_vector = tf.reduce_sum(context_embedding, axis=1)

        return context_vector


##### This uses mult Attention #####
class Attention_mult(Layer):
    def __init__(self):  # This layer is just to start defining the layer
        super(Attention_mult, self).__init__()

    def build(self, input_shape):  # This layer defines the shape of the weights and bias
        self.sequence_length = input_shape[0][1]  # the number of words (max_len)
        self.output_dimensions = input_shape[0][2]  # output dim [hidden vec dimensions] = H
        self.W = tf.keras.layers.Dense(self.output_dimensions)
        self.W.build(input_shape=(None, None, self.output_dimensions))  # (B, 1, H)
        super(Attention_mult, self).build(input_shape)

    def call(self, inputs):  # This is where the action happens
        # inputs is the input tensor
        target_hidden_state = inputs[1]  # (B, H)
        source_hidden_states = inputs[0]

        # h_t*W*h_s
        target_hidden_state = tf.expand_dims(input=target_hidden_state, axis=1)  # (B, 1, H)
        score = self.W(source_hidden_states)  # (B, S, H)
        score = target_hidden_state * score

        # softmax(S)
        weights = K.softmax(score, axis=1)  # This gives a_t; (B, S, H)

        # C_t= ∑(α_ts * h_s)
        context_embedding = source_hidden_states * weights  # (B, S, H)
        context_vector = tf.reduce_sum(context_embedding, axis=1)

        return context_vector
