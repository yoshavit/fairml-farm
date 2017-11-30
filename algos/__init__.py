from algos.baselines import SimpleNN, ParityNN, AdversariallyCensoredNN

# Add all new classifier types to this list!
classifier_types = [SimpleNN, ParityNN, AdversariallyCensoredNN]

def construct_classifier(hparams, sess=None, savefile=None):
    for c_type in classifier_types:
        if c_type.name == hparams["classifier_type"]:
            classifier = c_type(sess=sess)
            classifier.build(hparams=hparams)
            if savefile is not None:
                classifier.load_model(savefile)
            return classifier
    else:
        error_msg = "Invalid 'classifier_type' hparam; must be one of {}".format(
            [c_type.name for c_type in classifier_types])
        raise ValueError(error_msg)
