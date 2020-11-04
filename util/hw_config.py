def configure_gpu_cpu(run_gpu, gpu_allocation):
    """
    Takes care of some extra options when using GPU. It also give user control
    GPU percentage allocation.
    RUN_GPU: boolean, if GPU to be used set to true. If False it will force code
    to run on CPU
    GPU_ALLOCATION: decimal, set to the desire percentage i.e., 50= 50%
    """
    # Extra imports to set GPU options
    import tensorflow as tf
    from keras import backend as k
    import os
    # To force code to run on cpu
    if not run_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if run_gpu and gpu_allocation !=100:
        # TensorFlow config
        config = tf.ConfigProto()

        # Allocate memory as-needed
        config.gpu_options.allow_growth = True

        # Allocate GPU memory based on user input USE_GPU
        config.gpu_options.per_process_gpu_memory_fraction = gpu_allocation / 100

        # Create a session with the above specified options
        k.tensorflow_backend.set_session(tf.Session(config=config))
