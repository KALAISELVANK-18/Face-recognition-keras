from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace
import tensorflow as tf



#network memory limits

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



#generation of the images
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator =train_datagen.flow_from_directory(r"C:\Users\kalai\Desktop\projectkalai",target_size=(224,224),color_mode='rgb',batch_size=32,class_mode='categorical',shuffle=True)

train_generator.class_indices.values()
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())

base_model = VGGFace(include_top=False,model='vgg16',input_shape=(224, 224, 3))

base_model.summary()

print(len(base_model.layers))

x=base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)


# final layer with softmax activation
output = Dense(NO_CLASSES, activation='softmax')(x)
print(output)

model = Model(inputs = base_model.inputs, outputs = output)
model.build((224, 224, 3))
print(len(model.layers))



# don't train the first 19 layers - 0..18
for layer in model.layers[:19]:

    layer.trainable = False

# train the rest of the layers - 19 onwards
for layer in model.layers[19:]:
    layer.trainable = True

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_generator,batch_size = 1,verbose = 1,epochs = 20)
model.save('transfer_learning_trained' + '_face_cnn_model.h5')

print("the process is finished and the model is trained........")