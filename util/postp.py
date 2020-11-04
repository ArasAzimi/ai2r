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
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')

    print(">ia> Saving plot(s): {}".format(plot_path))
    plt.savefig(plot_path)


def save_model(model, model_path, lb, eval_report, CONFIG):
    """
    * Saves model and labels in pickle format
    * Saves the model summary and hyper parameters in .txt file
    """
    import pickle
    # save the model and label binarizer to disk
    print(">ia> Saving model: {}".format(model_path))
    model_labels = model_path + "_lbls.pickle"
    model_ = model_path + '.model'
    model_summary = model_path + '_summary.txt'
    model.save(model_)
    with open(model_labels, 'wb') as f:
        f.write(pickle.dumps(lb))
    f.close()

    with open(model_summary, 'w') as f:
        # Pass the file handle in as a lambda function to make it callable
        f.write("------ Model Summary ------\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("------ Model HyperParameters------\n")
        f.write("test_size: " + str(CONFIG['train']['test_size']) + '\n')
        f.write("learning_rate: " + str(CONFIG['train']['learning_rate']) + '\n')
        f.write("epochs: " + str(CONFIG['train']['epochs']) + '\n')
        f.write("batch_size: " + str(CONFIG['train']['batch_size']) + '\n')

        f.write("------ Model Evaluation Report------\n")
        f.write(eval_report)
    f.close()
