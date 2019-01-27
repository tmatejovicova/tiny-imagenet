import sys
sys.path.append('.')

import models
import train_test

models = models.get_models()

model_name = 'vgg'
model = models[model_name]
train_test.train_test(model, model_name, batch_size = 100, augment = True, epochs = 50)

model_name = 'conv_simple'
model = models[model_name]
train_test.train_test(model, model_name, batch_size = 100, augment = True, epochs = 50)

model_name = 'one_hidden'
model = models[model_name]
train_test.train_test(model, model_name, batch_size = 100, augment = True, epochs = 50)

model_name = 'log_reg'
model = models[model_name]
train_test.train_test(model, model_name, batch_size = 100, augment = True, epochs = 50)
