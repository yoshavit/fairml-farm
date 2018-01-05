from algos.nn_baselines import SimpleNN, ParityNN
from algos.nn_adv_censor import AdversariallyCensoredNN
# Import new classifier classes

# Add all new classifier types to this list
classifier_types = [SimpleNN, ParityNN, AdversariallyCensoredNN]

def construct_classifier(hparams, sess=None, loaddir=None):
    for c_type in classifier_types:
        if c_type.name == hparams["classifier_type"]:
            classifier = c_type(sess=sess)
            classifier.build(hparams=hparams)
            if loaddir is not None:
                classifier.load_model(loaddir)
            return classifier
    else:
        error_msg = "Invalid 'classifier_type' hparam; must be one of {}".format(
            [c_type.name for c_type in classifier_types])
        raise ValueError(error_msg)
