def plot_save_results(model_fit, model_path):
    """
    Plotting model's train/test loss/accuracy
    """
    import matplotlib
    import matplotlib.pyplot as plt

    plot_path = model_path + "_plt.png"

    plt.figure()
    plt.plot(model_fit.history["loss"], label="train_loss")
    plt.plot(model_fit.history["val_loss"], label="val_loss")
    plt.plot(model_fit.history["acc"], label="train_acc")
    plt.plot(model_fit.history["val_acc"], label="val_acc")
    plt.title("Train loss/accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc' ], loc='upper right')

    print(">ia> Saving plot(s): {}".format(plot_path))
    plt.savefig(plot_path)
