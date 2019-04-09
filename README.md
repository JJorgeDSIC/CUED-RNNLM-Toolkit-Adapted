# CUED-RNNLM-Toolkit-Adapted
Improving CUED-RNN Toolkit with new functionality

Modifications:
------------

* cued-rnnlm.v1.1: Fixed bug word max. length (fileops.*)
* cued-rnnlm.v1.1.evaloncpu: Setting lognormconst during training if criterion NCE or VR also when getting perplexity (calppl). Including an option to replace RNNLM prob per n-gram prob if a word is OOV.
