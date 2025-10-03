# import tensorflow.compat.v1 as tf
# from google.protobuf import text_format

# import _locale
# _locale.setlocale(_locale.LC_NUMERIC, 'en_US.UTF-8')


# graph_filename = "./frozen_models/graph_v1.pb"


# with tf.io.gfile.GFile(graph_filename, "rb") as f:
#     graph_def = tf.get_default_graph().as_graph_def()
#     graph_str = f.read()
#     graph_def.ParseFromString(f.read())
#     # with tf.Session(graph=graph_def) as persisted_sess:
#     #         persisted_sess.graph.as_default()
#     #         # tf.import_graph_def(graph_def, name='')
#     #         print("map variables")

#             # persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
#             # tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)

#             # init_op = tf.get_default_graph().get_operation_by_name("init")

#             # persisted_sess.run(init_op)

#             # saver = tf.train.Saver(tf.all_variables()) # 'Saver' misnomer! Better: Persister!
#             # print("load data")
#             # saver.restore(persisted_sess, "./out/build/x64-debug/checkpoints/checkpoint_1.data-00000-of-00001")  # now OK
#             # # print(persisted_result.eval())
#             # print("DONE")


#     g = tf.get_default_graph()

#     ops = g.get_operations()

#     for op in ops:
#         print(op.name, op.type)




# # Minimal code to successfully export tf modelimport time
# import time
# import os
# import tensorflow.compat.v1 as tf

# trained_checkpoint_prefix = "./out/build/x64-debug/checkpoints/checkpoint_1.data-00000-of-00001"
# export_dir = os.path.join('models', time.strftime("%Y%m%d-%H%M%S"))
# loaded_graph = tf.Graph()
# with tf.Session(graph=loaded_graph) as sess:
#     # Restore from checkpoint
#     loader = tf.train.import_meta_graph(trained_checkpoint_prefix)
#     loader.restore(sess, trained_checkpoint_prefix)
    
#     # Export checkpoint to SavedModel
#     builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
#     builder.add_meta_graph_and_variables(sess,
#                                          [tf.saved_model.tag_constants.TRAINING],
#                                          strip_default_attrs=True)
#     builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
#     builder.save()


import tensorflow as tf

imported = tf.saved_model.load("saved_model", tags=[])

print(imported)

f = imported.signatures["serving_default"]
print(f(x=tf.constant([[1.]])))