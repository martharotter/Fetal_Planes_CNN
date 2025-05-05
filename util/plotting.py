import matplotlib.pyplot as plt

def plot_training(train_losses, train_accs, val_losses, val_accs, model_name="", return_fig=True):
        """
        Plot losses and accuracy over the training process.

        train_losses (list): List of training losses over training.
        train_accs (list): List of training accuracies over training.
        val_losses (list): List of validation losses over training.
        val_accs (list):List of validation accuracies over training.
        model_name (str): Name of model as a string. 
        return_fig (Boolean): Whether to return figure or not. 
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        ax1.plot(val_losses, label='Validation loss')
        ax1.plot(train_losses, label="Training loss")
        ax1.set_title('Loss over training for {}'.format(model_name), fontsize=20)
        ax1.set_xlabel("epoch",fontsize=18)
        ax1.set_ylabel("loss",fontsize=18)
        ax1.legend()

        ax2.plot(val_accs, label='Validation accuracy')
        ax2.plot(train_accs, label='Training accuracy')
        ax2.set_title('Accuracy over training for {}'.format(model_name), fontsize=20)
        ax2.set_xlabel("epoch",fontsize=18)
        ax2.set_ylabel("accuracy",fontsize=18)
        ax2.legend()

        fig.tight_layout() 
        if return_fig:
                return fig

def plot_values_from_history(history, run_id, log):
        """
        Plot losses and accuracy over the training process.
        history (keras history object): History object from training.
        run_id (str): ID of run.
        log (logger): Logger object.
        """
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='validation accuracy')
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        log.info(f"Ending training run {run_id}")
