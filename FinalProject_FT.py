from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import time

model = load_model('FinalProject_ep150.h5')

train_dir='/home/jgh1320/PycharmProjects/test1/Final_Project/Img_Data/train'
test_dir='/home/jgh1320/PycharmProjects/test1/Final_Project/Img_Data/test'
validation_dir='/home/jgh1320/PycharmProjects/test1/Final_Project/Img_Data/val'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(400, 400),
    batch_size=30,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(400, 400),
    batch_size=30,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(400, 400),
    batch_size=30,
    class_mode='binary')

conv_base = model.layers[0]

for layer in conv_base.layers:
    if layer.name.startswith('block2'):
        layer.trainable = True
model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# main loop without cross-validation
starttime = time.time()
num_epochs = 200
history = model.fit_generator(train_generator,
                              epochs=num_epochs, steps_per_epoch=100,
                              validation_data=validation_generator, validation_steps=50)

model.save('FinalProject_FT_ep200.h5')

# evaluation

train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_loss', train_loss)
print('train_acc', train_acc)
print('test_loss', test_loss)
print('test_acc', test_acc)
print("elapsed time(in sec): ", time.time() - starttime)


# visualization

def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


plot_loss(history)
plt.savefig('loss_FT_re.png')
plt.clf()
plot_acc(history)
plt.savefig('accuracy_FT_re.png')
