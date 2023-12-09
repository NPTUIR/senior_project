def get_LPIPS_distance(true, fake):
    import tensorflow.compat.v1 as tf  # 用tf1.x的功能
    tf.disable_v2_behavior()  # 禁用2.x的功能
    from utils.lpips import lpips_tf
    # true&fake為np array型態 shape=(batch,w,h,channel)
    image0_ph = tf.convert_to_tensor(true, dtype=tf.float32, name="my_tensor")
    image1_ph = tf.convert_to_tensor(fake, dtype=tf.float32, name="my_tensor")
    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

    with tf.Session() as session:  # tf.Session是tf 1.x版本限定
        distance = session.run(distance_t, feed_dict={image0_ph: true, image1_ph: fake})
    return distance


