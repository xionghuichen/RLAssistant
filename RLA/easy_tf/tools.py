from RLA.easy_log.tester import tester


def run_with_metadata(sess, ops, feed, name):
    ts = tester.time_step_holder.get_time()
    tag = name + '-' + str(ts % 100)
    if tag in tester.metadata_list:
        # logger.warn("[WARN] repeat use runmeta data {}".format(tag))
        rets = sess.run(ops, feed_dict=feed)
    else:
        import tensorflow as tf
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        rets = sess.run(ops, feed_dict=feed,
                        options=run_options, run_metadata=run_metadata)
        tester.writer.add_run_metadata(run_metadata, tag)
        tester.metadata_list.append(tag)
    return rets