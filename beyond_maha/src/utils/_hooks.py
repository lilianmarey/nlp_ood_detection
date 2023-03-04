# Hook functions to get latent representation of data

def features_hook_0(model, inp, output):
    global feat_0
    feat_0 = output


def features_hook_1(model, inp, output):
    global feat_1
    feat_1 = output


def features_hook_2(model, inp, output):
    global feat_2
    feat_2 = output


def features_hook_3(model, inp, output):
    global feat_3
    feat_3 = output


def features_hook_4(model, inp, output):
    global feat_4
    feat_4 = output


def features_hook_5(model, inp, output):
    global feat_5
    feat_5 = output


def features_hook_6(model, inp, output):
    global feat_6
    feat_6 = output


def features_hook_7(model, inp, output):
    global feat_7
    feat_7 = output


def features_hook_8(model, inp, output):
    global feat_8
    feat_8 = output


def features_hook_9(model, inp, output):
    global feat_9
    feat_9 = output


def features_hook_10(model, inp, output):
    global feat_10
    feat_10 = output


def features_hook_11(model, inp, output):
    global feat_11
    feat_11 = output

# Create feature hook function for getting latent representation
features_hooks = [
                    features_hook_0, features_hook_1, features_hook_2,
                    features_hook_3, features_hook_4, features_hook_5,
                    features_hook_6, features_hook_7, features_hook_8,
                    features_hook_9, features_hook_10, features_hook_11
                ]
