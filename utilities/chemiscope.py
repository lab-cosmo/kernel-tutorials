# -*- coding: utf-8 -*-
'''
Generate JSON input files for the default chemiscope visualizer.
'''
import numpy as np


def _typetransform(data):
    if isinstance(data[0], str):
        return list(map(str, data))
    elif isinstance(data[0], bytes):
        return list(map(lambda u: u.decode('utf8'), data))
    else:
        try:
            return [float(value) for value in data]
        except ValueError:
            raise Exception('unsupported type in property')


def _linearize(name, values):
    '''
    Transform 2D arrays in multiple 1D arrays, converting types to fit json as
    needed.
    '''
    assert isinstance(values, np.ndarray)
    data = {}

    if len(values.shape) == 1:
        data[name] = _typetransform(values)
    elif len(values.shape) == 2:
        for i in range(values.shape[1]):
            data[f'{name}[{i + 1}]'] = _typetransform(values[:, i])
    else:
        raise Exception('unsupported ndarray property')

    return data


def _frame_to_json(frame):
    data = {}
    data['names'] = list(frame.symbols)
    data['x'] = [float(value) for value in frame.positions[:, 0]]
    data['y'] = [float(value) for value in frame.positions[:, 1]]
    data['z'] = [float(value) for value in frame.positions[:, 2]]

    if (frame.cell.lengths() != [0.0, 0.0, 0.0]).all():
        data['cell'] = list(np.concatenate(frame.cell))

    return data


def _generate_environments(frames, cutoff):
    environments = []
    for frame_id, frame in enumerate(frames):
        for center in range(len(frame)):
            environments.append({
                'structure': frame_id,
                'center': center,
                'cutoff': cutoff,
            })
    return environments


def chemiscope_input(name, frames, projection, prediction, property, property_name="", cutoff=None):
    '''
    Get a dictionary which can be saved as JSON and used as input data for the
    chemiscope visualizer (https://chemiscope.org).

    :param str name: name of the dataset
    :param list frames: list of `ase.Atoms`_ objects containing all the
                        structures
    :param array projection: projection of the structural descriptor in latent
                             space
    :param array prediction: predicted values for the properties for all
                             environments in the frames
    :param array property: actual value for properties for all environments in
                           the frames
    :param str property_name: name of the property being considered
    :param float cutoff: optional. If present, will be used to generate
                         atom-centered environments

    .. _`ase.Atoms`: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    '''

    data = {
        'meta': {
            'name': name,
        }
    }

    projection = np.asarray(projection)
    prediction = np.asarray(prediction)
    property = np.asarray(property)

    if not property_name:
        property_name = "property"

    assert projection.shape[0] == prediction.shape[0]
    assert projection.shape[0] == property.shape[0]

    n_atoms = sum(len(f) for f in frames)

    if projection.shape[0] == len(frames):
        target = 'structure'
    elif projection.shape[0] == n_atoms:
        target = 'atom'
    else:
        raise Exception(
            "the number of features do not match the number of environments"
        )

    error = np.sqrt((property - prediction) ** 2)
    properties = {}
    for name, values in _linearize("projection", projection).items():
        properties[name] = {"target": target, "values": values}

    for name, values in _linearize(property_name, property).items():
        properties[name] = {"target": target, "values": values}

    for name, values in _linearize("predicted {}".format(property_name), prediction).items():
        properties[name] = {"target": target, "values": values}

    for name, values in _linearize("{} error".format(property_name), error).items():
        properties[name] = {"target": target, "values": values}

    data['properties'] = properties
    data['structures'] = [_frame_to_json(frame) for frame in frames]

    if cutoff is not None:
        data['environments'] = _generate_environments(frames, cutoff)

    return data
