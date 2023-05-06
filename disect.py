from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
model = load_model('mobilenet_7.h5')
model.summary()

input = Input(shape=(224,224,3), dtype=float)
x = input
for layer in model.layers[1:-3]:
    x = layer(x)

backbone = Model(input, x)
backbone.compile()
backbone.save('mobilenet_7_backbone.h5')
