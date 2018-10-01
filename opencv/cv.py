import cv2 as cv
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

t1 = mnist.train.images[0]

ori_img = cv.imread('../1.jpg')
gray_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)

print(gray_img.shape)

cv.namedWindow('Image')
# cv.imshow('img', t1.reshape(28, 28))
cv.imshow('origin', ori_img)
cv.waitKey()

cv.imshow('gray', gray_img)
cv.waitKey()

small = cv.resize(gray_img, (28, 28))
cv.imshow('small', 1- small / 255)
cv.waitKey()
cv.destroyAllWindows()

# cv.imwrite('2.28.png', img=small)
# print('保存成功')
