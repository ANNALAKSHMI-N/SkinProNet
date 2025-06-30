from keras.layers import (
    Input,
    Conv1D,
    Dense,
    Lambda,
    Dot,
    Activation,
    Layer,
    Concatenate,
    LSTM,
)
from keras.models import Model
from keras import backend as K
from keras.optimizers.optimizer_v2.rmsprop import RMSProp
from keras.utils.vis_utils import plot_model


class AttentionLayer(Layer):
    def __init__(self, units=128, **kwargs):
        print("[INFO] Building Attention Layer")
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope("atttention"):
            self.attention_score_vec = Dense(
                input_dim, use_bias=False, name="attention_score_vec"
            )
            self.h_t = Lambda(
                lambda x: x[:, -1, :],
                output_shape=(input_dim,),
                name="last_hidden_state",
            )
            self.attention_score = Dot(axes=[1, 2], name="attention_score")
            self.attention_weight = Activation("softmax", name="attention_weight")
            self.context_vector = Dot(axes=[1, 1], name="context_vector")
            self.attention_output = Concatenate(name="attention_output")
            self.attention_vector = Dense(
                self.units, use_bias=False, activation="tanh", name="attention_vector"
            )
            super(AttentionLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        return super(AttentionLayer, self).__call__(inputs, training, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        score_first_part = self.attention_score_vec(inputs)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config


def buildACRNN(shape, out):
    print("[INFO] Building Attention-Covolutional RNN")
    inputs = Input(shape=shape)
    cnn = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(inputs)
    lstm = LSTM(64, return_sequences=True)(cnn)
    hopfield = AttentionLayer(units=16)(lstm)
    outputs = Dense(out, activation="softmax")(hopfield)
    model = Model(inputs=[inputs], outputs=[outputs], name="acrnn")
    print("[INFO] Compiling Model Using Root Mean Square Optimizer")
    opt = RMSProp(learning_rate=0.00016)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    plot_model(model, to_file="acrnn_model.png", show_shapes=True)
    return model
