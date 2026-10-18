"""Explicit input-shape rebinding for an unused terminal coordinate.

Author H5 declares 6001 but released CSV/notebook has 6000 coordinates.
With valid kernel-17 convolution and valid pool-2, the additional final
convolution position is discarded. Rebinding preserves every weight tensor.
Never insert a target label or fabricate an extra spectral coordinate.
"""
import hashlib
import json
import tf_keras as keras


def weight_digest(model):
    h = hashlib.sha256()
    for value in model.get_weights():
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def load_spectrum_model(path):
    author = keras.models.load_model(path, compile=False)
    assert tuple(author.input_shape[1:]) == (6001, 1)
    convolution = author.layers[0]
    pool = author.layers[3]
    assert isinstance(convolution, keras.layers.Conv1D)
    assert convolution.kernel_size == (17,) and convolution.strides == (1,)
    assert convolution.padding == 'valid' and convolution.dilation_rate == (1,)
    assert isinstance(pool, keras.layers.MaxPooling1D)
    assert pool.pool_size == (2,) and pool.strides == (2,) and pool.padding == 'valid'
    config = author.get_config()
    assert config['layers'][0]['class_name'] == 'InputLayer'
    config['layers'][0]['config']['batch_input_shape'] = [None, 6000, 1]
    config['layers'][1]['config']['batch_input_shape'] = [None, 6000, 1]
    rebound = keras.Sequential.from_config(config)
    rebound.set_weights(author.get_weights())
    assert weight_digest(rebound) == weight_digest(author)
    assert rebound.count_params() == author.count_params()
    return rebound
