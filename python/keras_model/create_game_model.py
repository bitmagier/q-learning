import tensorflow as tf
from keras import layers, models


class GameModel1(models.Sequential):
    def __init__(self, *args, **kwargs):
        super(GameModel1, self).__init__(*args, **kwargs)
        # TODO dimensions
        self.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.summary()

    @tf.function
    def call(self, inputs):
        output = self(inputs)
        return output

    @tf.function
    def train(self, train_data):
        loss = self.train_step(train_data)['loss']
        return loss


model = GameModel1()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




# TODO define a training mini-batch function
# e.g.
# - High level:
#   - input: last 5 images of the game
#   - output: action values: value vector for possible actions (left, none, right) [0..1]
# - steps:
#   - make a copy of the model
#   - run a min-batch (series of training data [input,target]
#           against the old model, but train the new one
#   - replace old model by the new model
#   - return stats (loss, etc)

