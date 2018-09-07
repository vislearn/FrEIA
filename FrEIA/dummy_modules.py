'''Dummy modules, used for testing the ReversibleGraphNet. Probably not useful
for anything else.'''


class dummy_data:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims


class dummy_module:
    def __init__(self, dims_in, **args):
        self.dims = dims_in

    @staticmethod
    def output_dims(input_dims):
        return input_dims

    def __call__(self, inp):
        return self.output_dims(self.dims)


class dummy_2split(dummy_module):
    def __init__(self, dims_in, **args):
        self.dims = dims_in

    @staticmethod
    def output_dims(input_dims):
        ch_in = input_dims[0][0]
        ch, w, h = (ch_in//2,
                    input_dims[0][1],
                    input_dims[0][2])
        assert len(input_dims) == 1
        return [(ch, w, h), (ch_in-ch, w, h)]


class dummy_2merge(dummy_module):
    def __init__(self, dims_in, **args):
        self.dims = dims_in

    @staticmethod
    def output_dims(input_dims):
        assert len(input_dims) == 2

        ch_tot = input_dims[0][0] + input_dims[1][0]
        return [(ch_tot, input_dims[0][1], input_dims[0][2])]


class dummy_mux(dummy_module):
    def __init__(self, dims_in, **args):
        self.dims = dims_in

    @staticmethod
    def output_dims(input_dims):
        ch, w, h = (input_dims[0][0]*4,
                    input_dims[0][1]//2,
                    input_dims[0][2]//2)
        assert len(input_dims) == 1
        assert ch*w*h == input_dims[0][0] * input_dims[0][1] * input_dims[0][2]
        return [(ch, w, h)]
