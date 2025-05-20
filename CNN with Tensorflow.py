import tensorflow as tf
import matplotlib.pyplot as plt
import os
import warnings
import time
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
warnings.filterwarnings('ignore')

tf.compat.v1.disable_eager_execution()

# 数据所在文件夹
base_dir = 'E:\\Software\\Python\\pycharm\\AI基础\\路面识别'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

BATCH_SIZE = 32  # 批量处理时每次处理的数据量大小
IMG_SIZE = (160, 160)  # 图像统一大小

# 创建dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            validation_split=0.2,
                                                            subset="training",
                                                            seed=123)

valid_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            validation_split=0.2,
                                                            subset="validation",
                                                            seed=123)

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=False,
                                                           image_size=IMG_SIZE)

# 取分类名称
class_names = train_dataset.class_names
print(class_names)

# 开辟一个固定内存，和batch_size一致
train_dataset = train_dataset.prefetch(buffer_size=BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(buffer_size=BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=BATCH_SIZE)


# 数据增强
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')  # 水平翻转
])


# 获取预训练模型对输入的预处理方法
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 数据标准化
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

# 创建预训练模型
IMG_SIZE = (160, 160, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE,
                                               alpha=1.0,  # 同比例改变每层滤波器个数
                                               include_top=False,  # 是否包含顶层的全连接层
                                               weights='imagenet'
                                               )
'''
# ResNet152V2
base_model = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=(160, 160, 3))

# ResNet50
base_model = tf.keras.applications.resnet.ResNet50(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(160, 160, 3))                                               

# DenseNet121
base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=(160, 160, 3))

# DenseNet201
base_model = tf.keras.applications.densenet.DenseNet201(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=(160, 160, 3))

# Xception
base_model = tf.keras.applications.xception.Xception(input_shape=IMG_SIZE,
                                                     include_top=False,
                                                     weights='imagenet')

# NASNetLarge
base_model = tf.keras.applications.nasnet.NASNetLarge(include_top=False,
                                                      input_shape=IMG_SIZE,
                                                      weights='imagenet')

# NASNetMobile
base_model = tf.keras.applications.nasnet.NASNetMobile(include_top=False,
                                                       input_shape=IMG_SIZE,
                                                       weights='imagenet')
'''

# 训练部分
inputs = tf.keras.Input(shape=(160, 160, 3))
# 数据增强
x = data_augmentation(inputs)
# 数据预处理
x = preprocess_input(x)
# 模型
x = base_model(x, training=False)  # 参数不变化
# 全局池化
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# 隐藏层1
x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
# 隐藏层2
x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
# Dropout
x = tf.keras.layers.Dropout(0.4)(x)
# 输出层
outputs = tf.keras.layers.Dense(31, activation='softmax')(x)
# 整体封装
model = tf.keras.Model(inputs, outputs)


# Fine tuning
base_model.trainable = True

# 保留前fine_tune_at层的参数
fine_tune_at = 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at+1:]:
    layer.trainable = True

start = time.perf_counter()

# 输出模型情况
model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

# 迭代式微调训练
history_fine = model.fit_generator(train_dataset,
                                   epochs=35,
                                   validation_data=valid_dataset)

# 输出损失曲线
plt.plot(history_fine.history['loss'])
plt.plot(history_fine.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# 输出准确率曲线
plt.plot(history_fine.history['accuracy'])
plt.plot(history_fine.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# 查看测试集结果
loss, accuracy = model.evaluate(test_dataset)
print("Loss: {:.4f}".format(loss))
print("Accuracy: {:.4f}".format(accuracy))

model.save('mobilenet_v2.h5')

end = time.perf_counter()
print("运行时间：", (end - start))


# 绘制混淆矩阵
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Greens, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(31, 31))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, truelabel, labels):
    predictions = model.predict(test_dataset)
    predictions = predictions.argmax(axis=-1)
    for i in range(606):
        predictions[i] = predictions[i] + 1
        if truelabel[i] >= 24:
            predictions[i] = predictions[i] + 1
        if truelabel[i] >= 30:
            predictions[i] = predictions[i] + 1
        if truelabel[i] >= 31:
            predictions[i] = predictions[i] + 1
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    # plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix')


# 创建正确输出标签
truelabel = []

for i in class_names:
    if i == '15':
        for j in range(19):
            x = int(i)
            truelabel.append(x)

    elif i == '18':
        for j in range(14):
            x = int(i)
            truelabel.append(x)

    elif i == '20':
        for j in range(17):
            x = int(i)
            truelabel.append(x)

    elif i == '21':
        for j in range(18):
            x = int(i)
            truelabel.append(x)

    elif i == '26':
        for j in range(19):
            x = int(i)
            truelabel.append(x)

    elif i == '33':
        for j in range(19):
            x = int(i)
            truelabel.append(x)

    else:
        for j in range(20):
            x = int(i)
            truelabel.append(x)

lable = []
for i in range(1, 35):
    x = str(i)
    lable.append(x)

plot_confuse(model, truelabel, lable)
