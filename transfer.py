from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.vgg16 import VGG16




train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,  # Randomly rotate images by up to 40 degrees
                                   width_shift_range=0.2,
                                   # Randomly shift images horizontally by up to 20% of the width
                                   height_shift_range=0.2,
                                   # Randomly shift images vertically by up to 20% of the height
                                   shear_range=0.2,  # Randomly apply shearing transformations
                                   zoom_range=0.2,  # Randomly zoom in on images
                                   horizontal_flip=True,  # Randomly flip images horizontally
                                   fill_mode='nearest'
                                   )

train_generator =train_datagen.flow_from_directory(r"C:\train",target_size=(224,224),color_mode='rgb',batch_size=128,class_mode='categorical',shuffle=True)

train_generator.class_indices.values()
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())
base_model = VGG16(include_top=False,input_shape=(224, 224, 3),weights='imagenet')
base_model.summary()

print(len(base_model.layers))

x=base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(200, activation='relu')(x)
x = Dense(170, activation='relu')(x)



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


model.fit(train_generator,batch_size = 128,verbose = 1,epochs = 50)
model.save('main_' + 'model.h5')

print("the process is finished and the model is trained........")