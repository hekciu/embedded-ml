import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()

GRAPH_FILENAME = 'frozen_models/graph_v1.pb'
CHECKPOINT_PREFIX = 'out/build/x64-debug/checkpoints/checkpoint_1'
EXPORT_DIR = 'exported_savedmodel'

graph = tf.Graph()
with graph.as_default():
    with tf.io.gfile.GFile(GRAPH_FILENAME, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

with tf.Session(graph=graph) as sess:
    try:
        restore_op = graph.get_operation_by_name("save/restore_all")
        checkpoint_path_tensor = graph.get_tensor_by_name("save/Const:0")
    except KeyError:
        print("Error: Could not find the 'save/restore_all' op in the graph.")
        print("Ensure the original graph was created with a tf.train.Saver().")
        exit()

    sess.run(restore_op, feed_dict={checkpoint_path_tensor: CHECKPOINT_PREFIX})
    print(f"Successfully restored weights from '{CHECKPOINT_PREFIX}'")

    input_tensor = graph.get_tensor_by_name("input:0")
    output_tensor = graph.get_tensor_by_name("output:0")

    builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)

    signature = tf.saved_model.predict_signature_def(
        inputs={'input': input_tensor},
        outputs={'output': output_tensor}
    )

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'serving_default': signature}
    )
    builder.save()
    print(f"Model successfully exported to '{EXPORT_DIR}'")