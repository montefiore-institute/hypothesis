r"""Summary objects at the end of training procedures."""

import pickle



class TrainingSummary:

    def __init__(self,
        model_best,
        model_final,
        epochs,
        epoch_best,
        losses_train,
        losses_test=None,
        identifier=None):
        self.identifier = identifier
        self.epochs = epochs
        self.model_best = model_best
        self.model_final = model_final
        self.epoch_best = epoch_best
        self.losses_train = losses_train
        self.losses_test = losses_test

    def save(self, path):
        summary = {
            "identifier": self.identifier,
            "best_model": self.model_best,
            "final_model": self.model_final,
            "epochs": self.epochs,
            "best_epoch": self.epoch_best,
            "training_losses": self.losses_train,
            "testing_losses": self.losses_test}
        with open(path, "wb") as handle:
            pickle.dump(summary, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            summary = pickle.load(handle)
        self.identifier = summary["identifier"]
        self.model_best = summary["best_model"]
        self.model_final = summary["final_model"]
        self.epochs = summary["epochs"]
        self.epoch_best = summary["best_epoch"]
        self.losses_train = summary["training_losses"]
        self.losses_test = summary["testing_losses"]

    def test_losses_available(self):
        return self.losses_test is not None and len(self.losses_test) > 0

    def identifier_available(self):
        return self.identifier is not None

    def best_epoch(self):
        return self.epoch_best

    def best_model(self):
        return self.model_best

    def final_model(self):
        return self.model_final

    def testing_loss():
        return losses_test

    def training_loss():
        return losses_train

    def __str__(self):
        representation = ""
        if self.identifier_available():
            representation = "Identifier:\t\t{}\n".format(self.identifier)
        representation = representation + "Total epochs:\t\t{}\n".format(self.epochs) + \
          "Best training loss:\t{}\n".format(self.losses_train.min()) + \
          "Final training loss:\t{}".format(self.losses_train[-1])
        if self.test_losses_available():
            representation = representation + \
              "\nBest testing loss:\t{}\n".format(self.losses_test.min()) + \
              "Best test epoch:\t{}\n".format(self.epoch_best) + \
              "Final test loss:\t{}".format(self.losses_test[-1])

        return representation
