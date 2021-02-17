Module hypothesis.train.base
============================
Base definition of a neural network trainer with customizable hooks.

Classes
-------

`BaseTrainer(accelerator=device(type='cpu'), batch_size=4096, dataset_test=None, dataset_validate=None, dataset_train=None, epochs=100, pin_memory=True, shuffle=True, workers=4)`
:   

    ### Ancestors (in MRO)

    * hypothesis.engine.base.Procedure

    ### Descendants

    * hypothesis.train.ratio_estimation.base.RatioEstimatorTrainer

    ### Instance variables

    `accelerator`
    :

    `current_epoch`
    :

    `epochs`
    :

    `losses_test`
    :

    `losses_train`
    :

    `losses_validate`
    :

    ### Methods

    `fit(self)`
    :

    `test(self)`
    :

    `train(self)`
    :

    `validate(self)`
    :