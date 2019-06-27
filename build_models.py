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
    tf.compat.v1.saved_model.simple_save(
        tf.keras.backend.get_session(),
        export_path,
        inputs={"image_bytes": string_input},
        outputs={"predictions": output},
    )


architectures = [
        ("vgg16", tf.keras.applications.vgg16.VGG16),
        ("resnet50", tf.keras.applications.resnet50.ResNet50),
        ("inception_v3", tf.keras.applications.inception_v3.InceptionV3),
        ("xception", tf.keras.applications.xception.Xception),
        ("mobilenet", tf.keras.applications.mobilenet.MobileNet),
        ("mobilenet_v2", tf.keras.applications.mobilenet_v2.MobileNetV2),
        ]

config_lines = []
for name, cls in architectures:
    h5file = f"model_{name}.h5"
    build_model(cls, h5file)
    create_saved_model(h5file, f"models/{name}/1")

    config_lines.append("""
    config: {{
        name: '{name}'
        base_path: '/models/{name}'
        model_platform: 'tensorflow'
    }}""".format(name=name))

config = """model_config_list: {{
    {configs}

}}""".format(configs="\n".join(config_lines))

with open("ModelConfig.pbtext", "w") as outfile:
    outfile.write(config)
