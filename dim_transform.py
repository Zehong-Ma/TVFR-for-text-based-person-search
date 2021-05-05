import tensorflow as tf
x1=tf.constant([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                    [[11,12,13,14],[15,16,17,18],[19,20,21,22]]],dtype=tf.float32)
x2=tf.constant([[[11,12,13,14],[15,16,17,18],[19,20,21,22]],
                [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
                    ],dtype=tf.float32)
x2=tf.einsum('bgf->bfg',x1)
#x1= tf.einsum('gbf->bgf',x)
#x = tf.reduce_mean(tf.reduce_mean(tf.square(tf.subtract(x1,x2)),axis=-1),axis=-1)
x=tf.einsum('bgf,bfm->bgm',x1,x2)

c = tf.constant([1,2,3,4,5])
c1 = tf.tile(tf.expand_dims(c,-1),[1,4])
c2 = tf.reshape(c1,[5,4])
with tf.Session() as sess:
    print(sess.run(c2))
    