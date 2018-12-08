# -*- coding: utf-8 -*-

from __future__ import print_function

from datetime import datetime, timedelta
import functools
import itertools

import pytest
from numpy.random import randn

import numpy as np
import numpy.ma as ma
import numpy.ma.mrecords as mrecords

from pandas.core.dtypes.common import is_integer_dtype
from pandas.compat import (lmap, long, zip, range, lrange, lzip,
                           OrderedDict, is_platform_little_endian, PY3, PY36)
from pandas import compat
from pandas import (DataFrame, Index, Series, isna,
                    MultiIndex, Timedelta, Timestamp,
                    date_range, Categorical)
import pandas as pd
import pandas.util.testing as tm
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike

from pandas.tests.frame.common import TestData


MIXED_FLOAT_DTYPES = ['float16', 'float32', 'float64']
MIXED_INT_DTYPES = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
                    'int32', 'int64']


class TestDataFrameCreation(TestData):

    def test_creation_mixed(self):
        index, data = tm.getMixedTypeDict()

        indexed_frame = DataFrame.from_dict(data, orient=DataFrame.COLUMNS).set_index(index).build()  # noqa
        unindexed_frame = DataFrame.from_dict(data, orient=DataFrame.COLUMNS).build()  # noqa

        assert self.mixed_frame['foo'].dtype == np.object_

    def test_creation_cast_failure(self):
        foo = DataFrame.from_dict({'a': ['a', 'b', 'c']}, orient=DataFrame.COLUMNS).astype(np.float64).build()
        assert foo['a'].dtype == object

        df = DataFrame.from_ndarray(np.ones((4, 2))).build()

        # this is ok
        df['foo'] = np.ones((4, 2)).tolist()

        # this is not ok
        pytest.raises(ValueError, df.__setitem__, tuple(['test']),
                      np.ones((4, 2)))

        # this is ok
        df['foo2'] = np.ones((4, 2)).tolist()

    def test_creation_dtype_copy(self):
        orig_df = DataFrame.from_dict({
            'col1': [1.],
            'col2': [2.],
            'col3': [3.]}, orient=DataFrame.COLUMNS).build()

        new_df = pd.DataFrame.from_dataframe(orig_df, copy=True).astype(float).build()

        new_df['col1'] = 200.
        assert orig_df['col1'][0] == 1.

    def test_creation_dtype_nocast_view(self):
        df = DataFrame.from_list([[1, 2]]).build()
        should_be_view = DataFrame.from_dataframe(df).astype(df[0].dtype).build()
        should_be_view[0][0] = 99
        assert df.values[0, 0] == 99

        should_be_view = DataFrame.from_ndarray(df.values).astype(df[0].dtype).build()
        should_be_view[0][0] = 97
        assert df.values[0, 0] == 97

    def test_creation_dtype_list_data(self):
        df = DataFrame.from_list([[1, '2'],
                        [None, 'a']]).astype(object).build()
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == '2'

    def test_creation_list_frames(self):
        result = DataFrame.from_list([DataFrame([])]).build()
        assert result.shape == (1, 0)

        result = DataFrame.from_list([DataFrame(dict(A=lrange(5)))]).build()
        assert isinstance(result.iloc[0, 0], DataFrame)

    def test_creation_dict(self):
        frame = DataFrame.from_dict({'col1': self.ts1,
                           'col2': self.ts2}, orient=DataFrame.COLUMNS).build()

        # col2 is padded with NaN
        assert len(self.ts1) == 30
        assert len(self.ts2) == 25

        tm.assert_series_equal(self.ts1, frame['col1'], check_names=False)

        exp = pd.Series(np.concatenate([[np.nan] * 5, self.ts2.values]),
                        index=self.ts1.index, name='col2')
        tm.assert_series_equal(exp, frame['col2'])

        # Length-one dict micro-optimization
        frame = DataFrame.from_dict({'A': {'1': 1, '2': 2}}, orient=DataFrame.COLUMNS).build()
        tm.assert_index_equal(frame.index, pd.Index(['1', '2']))

    def test_creation_dict_of_tuples(self):
        data = {'a': (1, 2, 3), 'b': (4, 5, 6)}

        result = DataFrame.from_dict(data, orient=DataFrame.COLUMNS).build()
        expected = DataFrame.from_dict({k: list(v) for k, v in compat.iteritems(data)}, orient=DataFrame.COLUMNS).build()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_creation_DataFrame(self):
        df = DataFrame.from_dataframe(self.frame).build()
        tm.assert_frame_equal(df, self.frame)

        df_casted = DataFrame.from_dataframe(self.frame).astype(np.int64).build()
        assert df_casted.values.dtype == np.int64

    def test_creation_list_of_lists(self):
        df = DataFrame.from_list(data=[[1, 'a'], [2, 'b']]).set_columns(["num", "str"]).build()
        assert is_integer_dtype(df['num'])
        assert df['str'].dtype == np.object_

    def test_creation_list_of_dicts(self):
        data = [OrderedDict([['a', 1.5], ['b', 3], ['c', 4], ['d', 6]]),
                OrderedDict([['a', 1.5], ['b', 3], ['d', 6]]),
                OrderedDict([['a', 1.5], ['d', 6]]),
                OrderedDict(),
                OrderedDict([['a', 1.5], ['b', 3], ['c', 4]]),
                OrderedDict([['b', 3], ['c', 4], ['d', 6]])]

        result = DataFrame.from_list(data).build()
        expected = DataFrame.from_dict(dict(zip(range(len(data)), data)),
                                       orient=DataFrame.INDEX).build()
        tm.assert_frame_equal(result, expected.reindex(result.index))

    def test_creation_orient(self):
        data_dict = self.mixed_frame.T._series
        recons = DataFrame.from_dict(data_dict, orient=DataFrame.INDEX).build()
        expected = self.mixed_frame.sort_index()
        tm.assert_frame_equal(recons, expected)

        # dict of sequence
        a = {'hi': [32, 3, 3],
             'there': [3, 5, 3]}
        rs = DataFrame.from_dict(a, orient=DataFrame.INDEX).build()
        xp = DataFrame.from_dict(a, orient=DataFrame.COLUMNS).build().T.reindex(list(a.keys()))
        tm.assert_frame_equal(rs, xp)

    def test_creation_ndarray_copy(self):
        df = DataFrame.from_ndarray(self.frame.values).build()

        self.frame.values[5] = 5
        assert (df.values[5] == 5).all()

        df = DataFrame.from_ndarray(self.frame.values, copy=True).build()
        self.frame.values[6] = 6
        assert not (df.values[6] == 6).all()

