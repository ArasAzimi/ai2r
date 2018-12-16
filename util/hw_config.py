def configure_gpu_cpu(RUN_GPU, GPU_ALLOCATION):
    """
    Takes care of some extra options when using GPU. It also give user control
    GPU percenatge allocation.
    RUN_GPU: boolian, if GPU to be used set to true. If False it will force code
    to run on CPU
    GPU_ALLOCATION: decimal, set to the desire percenatge i.e., 50= 50%
    """
    # Extra imports to set GPU options
    import tensorflow as tf
    from keras import backend as k
    import os
    # To force code to run on cpu
    if RUN_GPU==False:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if RUN_GPU and GPU_ALLOCATION !=100:
        # TensorFlow congif
        config = tf.ConfigProto()

        # Allocate memory as-needed
        config.gpu_options.allow_growth = True

        # Allocate GPU memory based on user input USE_GPU
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ALLOCATION/100

        # Create a session with the above specified options
        k.tensorflow_backend.set_session(tf.Session(config=config))
