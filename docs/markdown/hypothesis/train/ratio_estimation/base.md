Module hypothesis.train.ratio_estimation.base
=============================================

Classes
-------

`RatioEstimatorTrainer(estimator, optimizer, accelerator=device(type='cpu'), batch_size=4096, conservativeness=0.0, dataset_test=None, dataset_train=None, dataset_validate=None, epochs=1, logits=True, pin_memory=True, shuffle=True, show=False, workers=4)`
:   

    ### Ancestors (in MRO)

    * hypothesis.train.base.BaseTrainer
    * hypothesis.engine.base.Procedure

    ### Instance variables

    `best_estimator`
    :

    `best_state_dict`
    :

    `conservativeness`
    :

    `conservativenesses`
    :

    `estimator`
    :

    `optimizer`
    :

    `state_dict`
    :

    ### Methods

    `test(self)`
    :

    `train(self)`
    :

    `validate(self)`
    :