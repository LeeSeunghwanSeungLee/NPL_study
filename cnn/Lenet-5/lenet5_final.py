'''
   이승환승이
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model




mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalization

# additional channel(dimension)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_test.shape)

# generator
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Network
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(6, kernel_size=5, padding='same', activation='sigmoid')
    self.maxpool_1 = MaxPool2D()
    
    self.conv2 = Conv2D(16, kernel_size = 5, activation = 'sigmoid')
    self.maxpool_2 = MaxPool2D()

    self.flatten = Flatten()
    self.dense_1 = Dense(120, activation = 'sigmoid')
    self.dense_2 = Dense(84, activation = 'sigmoid')
    self.dense_3 = Dense(10, activation = 'softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.maxpool_1(x)
    x = self.conv2(x)
    x = self.maxpool_2(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dense_2(x)
    return self.dense_3(x)

model = MyModel()


# loss function + optimizer setting
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function

def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'epoch: {}, loss: {}, accuracy: {}, test_loss: {}, test_accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))


# model save
model.save_weights('lenet_model_tf_v.2.h5')

# model load
new_model = tf.keras.models.load_model('lenet_model_tf_v2.h5')

# summary
new_model.summary()