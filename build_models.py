#!/usr/bin/env python

import shutil
import os
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    default=False,
    help="Force overriding of the cached models, if they exist already",
)
args = parser.parse_args()


def build_model(arch, filename):
    if os.path.isfile(filename) and not args.force:
        return

    model = arch(weights="imagenet")
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.save(filename)


def create_saved_model(h5file, export_path):
    if os.path.isfile(os.path.join(export_path, "saved_model.pb")) and not args.force:
        return

    string_input = tf.compat.v1.placeholder(tf.string, shape=(None,))
    imgs_map = tf.map_fn(tf.image.decode_image, string_input, dtype=tf.uint8)

    imgs_map.set_shape((None, None, None, 3))
    imgs = tf.image.resize_images(imgs_map, [224, 224])
    imgs = tf.reshape(imgs, (-1, 224, 224, 3))
    img_float = tf.cast(imgs, dtype=tf.float32)

    # Imagenet preprocessing
    img_float /= 127.5
    img_float -= 1.0

    model = tf.keras.models.load_model(h5file)
    output = model(img_float)

    shutil.rmtree(export_path, ignore_errors=True)
    tf.saved_model.simple_save(
        tf.keras.backend.get_session(),
        export_path,
        inputs={"image_bytes": string_input},
        outputs={"predictions": output},
    )


vgg16_h5file = "model_vgg16.h5"
resnet_h5file = "model_resnet50.h5"

build_model(tf.keras.applications.vgg16.VGG16, vgg16_h5file)
build_model(tf.keras.applications.resnet50.ResNet50, resnet_h5file)

create_saved_model(vgg16_h5file, "models/vgg16/1")
create_saved_model(resnet_h5file, "models/resnet50/1")
