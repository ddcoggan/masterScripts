import pytest
import pandas as pd
import xarray as xr
from xarray import DataArray

from brainio_base.assemblies import DataAssembly, get_levels, gather_indexes, is_fastpath


def test_get_levels():
    assy = DataAssembly(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        coords={
            'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
            'down': ("a", [1, 1, 1, 1, 2, 2]),
            'sideways': ('b', ['x', 'y', 'z'])
        },
        dims=['a', 'b']
    )
    assert get_levels(assy) == ["up", "down"]


class TestSubclassing:
    def test_wrap_dataarray(self):
        da = DataArray(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            coords={
                'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                'down': ("a", [1, 1, 1, 1, 2, 2]),
                'sideways': ('b', ['x', 'y', 'z'])
            },
            dims=['a', 'b']
        )
        assert "up" in da.coords
        assert da["a"].variable.level_names is None
        da = gather_indexes(da)
        assert da.coords.variables["a"].level_names == ["up", "down"]
        assert da["a"].variable.level_names == ["up", "down"]
        da = DataArray(da)
        assert da.coords.variables["a"].level_names == ["up", "down"]
        assert da["up"] is not None

    def test_wrap_dataassembly(self):
        assy = DataAssembly(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            coords={
                'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                'down': ("a", [1, 1, 1, 1, 2, 2]),
                'sideways': ('b', ['x', 'y', 'z'])
            },
            dims=['a', 'b']
        )
        assert assy.coords.variables["a"].level_names == ["up", "down"]
        assert assy["a"].variable.level_names == ["up", "down"]
        da = DataArray(assy)
        assert da.coords.variables["a"].level_names == ["up", "down"]
        assert da["a"].variable.level_names == ["up", "down"]
        assert da["up"] is not None

    def test_reset_index(self):
        assy = DataAssembly(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            coords={
                'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                'down': ("a", [1, 1, 1, 1, 2, 2]),
                'sideways': ('b', ['x', 'y', 'z'])
            },
            dims=['a', 'b']
        )
        da = DataArray(assy)
        da = da.reset_index(["up", "down"])
        assert get_levels(da) == []

    def test_repr(self):
        assy = DataAssembly(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            coords={
                'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                'down': ("a", [1, 1, 1, 1, 2, 2]),
                'back': ('b', ['x', 'y', 'z']),
                'forth': ('b', [True, True, False])
            },
            dims=['a', 'b']
        )
        assy_repr = repr(assy)
        print(assy_repr)
        assert "up" in assy_repr
        assert "down" in assy_repr
        assert "back" in assy_repr
        assert "forth" in assy_repr

    def test_getitem(self):
        assy = DataAssembly(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            coords={
                'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                'down': ("a", [1, 1, 1, 1, 2, 2]),
                'sideways': ('b', ['x', 'y', 'z'])
            },
            dims=['a', 'b']
        )
        single = assy[0, 0]
        assert type(single) is type(assy)

    def test_is_fastpath(self):
        """In DataAssembly.__init__ we have to check whether fastpath is present in a set of arguments and true
        or truthy (as interpreted by DataArray.__init__), even if only positional arguments are passed.  """
        assert is_fastpath(0, None, None, None, None, None, True)
        assert is_fastpath(0, None, None, None, None, None, fastpath=True)
        assert is_fastpath(0, None, None, None, None, None, 1)
        assert is_fastpath(0, None, None, None, None, None, fastpath=1)
        assert not is_fastpath(0, None, None, None, None, None, False)
        assert not is_fastpath(0, None, None, None, None, None, 0)
        assert not is_fastpath(0, None, None, None, None, None, fastpath=False)
        assert not is_fastpath(0, None, None, None, None, None, fastpath=0)
        assert not is_fastpath(0, None, None, None, None, None)

    def test_fastpath_signature_change(self):
        """Make sure that xarray hasn't changed the signature of DataArray.__init__ (again),
        because is_fastpath assumes a fixed length.  """
        # If the number of parameters has decreased, this will error:
        d = xr.DataArray(0, None, None, None, None, None, False)
        # This should raise TypeError: "__init__() takes from 1 to 8 positional arguments but 9 were given"
        with pytest.raises(TypeError) as te:
            d = xr.DataArray(0, None, None, None, None, None, None, False)
        assert "but 9" in str(te.value)
        # If they move fastpath to another spot in the list I guess it's tough.

    def test_align(self):
        name = "foo",
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
        mi1 = pd.MultiIndex.from_arrays([
            ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta'],
            [1, 2, 3, 4, 5, 6]
        ], names=["up", "down"])
        mi2 = pd.MultiIndex.from_arrays([
            ['alpha', 'alpha', 'beta', 'beta', 'gamma', 'gamma'],
            [1, 2, 3, 4, 5, 6]
        ], names=["up", "down"])
        da1 = DataArray(
            name=name,
            data=data,
            coords={
                'a': mi1,
                'b': ['x', 'y', 'z']
            },
            dims=['a', 'b']
        )
        da2 = DataArray(
            name=name,
            data=data,
            coords={
                'a': mi2,
                'b': ['x', 'y', 'z']
            },
            dims=['a', 'b']
        )
        assert hasattr(da1, "up")
        assert da1.coords.variables["a"].level_names == ["up", "down"]
        assert da1["a"].variable.level_names == ["up", "down"]
        assert da1["up"] is not None
        aligned1, aligned2 = xr.align(da1, da2, join="outer")
        assert hasattr(aligned1, "up")
        assert aligned1.coords.variables["a"].level_names == ["up", "down"]
        assert aligned1["a"].variable.level_names == ["up", "down"]
        assert aligned1["up"] is not None
        assert hasattr(aligned2, "up")
        assert aligned2.coords.variables["a"].level_names == ["up", "down"]
        assert aligned2["a"].variable.level_names == ["up", "down"]
        assert aligned2["up"] is not None


class TestIndex:
    def test_single_element(self):
        d = DataAssembly([0], coords={'coordA': ('dim', [0]), 'coordB': ('dim', [1])}, dims=['dim'])
        d.sel(coordA=0)
        d.sel(coordB=1)

    def test_multi_elements(self):
        d = DataAssembly([0, 1, 2, 3, 4],
                         coords={'coordA': ('dim', [0, 1, 2, 3, 4]),
                                 'coordB': ('dim', [1, 2, 3, 4, 5])},
                         dims=['dim'])
        d.sel(coordA=0)
        d.sel(coordA=4)
        d.sel(coordB=1)
        d.sel(coordB=5)

    def test_incorrect_coord(self):
        d = DataAssembly([0], coords={'coordA': ('dim', [0]), 'coordB': ('dim', [1])}, dims=['dim'])
        with pytest.raises(KeyError):
            d.sel(coordA=1)
        with pytest.raises(KeyError):
            d.sel(coordB=0)


class TestMultiGroupby:
    def test_single_dimension(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': ['a', 'b'], 'b': ['x', 'y', 'z']}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean(...)
        assert g.equals(DataAssembly([2, 5], coords={'a': ['a', 'b']}, dims=['a']))

    def test_single_dimension_int(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': [1, 2], 'b': [3, 4, 5]}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean(...)
        assert g.equals(DataAssembly([2., 5.], coords={'a': [1, 2]}, dims=['a']))

    def test_single_coord(self):
        d = DataAssembly(
            data=[
                [ 0,  1,  2,  3,  4,  5,  6],
                [ 7,  8,  9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20]
            ],
            coords={
                "greek": ("a", ['alpha', 'beta', 'gamma']),
                "colors": ("a", ['red', 'green', 'blue']),
                "compass": ("b", ['north', 'south', 'east', 'west', 'northeast', 'southeast', 'southwest']),
                "integer": ("b", [0, 1, 2, 3, 4, 5, 6]),
            },
            dims=("a", "b")
        )
        g = d.multi_groupby(['greek']).mean(...)
        c = DataAssembly(
            data=[3, 10, 17],
            coords={'greek': ('greek', ['alpha', 'beta', 'gamma'])},
            dims=['greek']
        )
        assert g.equals(c)

    def test_single_dim_multi_coord(self):
        d = DataAssembly([1, 2, 3, 4, 5, 6],
                         coords={'a': ('multi_dim', ['a', 'a', 'a', 'a', 'a', 'a']),
                                 'b': ('multi_dim', ['a', 'a', 'a', 'b', 'b', 'b']),
                                 'c': ('multi_dim', ['a', 'b', 'c', 'd', 'e', 'f'])},
                         dims=['multi_dim'])
        g = d.multi_groupby(['a', 'b']).mean()
        assert g.equals(DataAssembly([2, 5],
                                     coords={'a': ('multi_dim', ['a', 'a']), 'b': ('multi_dim', ['a', 'b'])},
                                     dims=['multi_dim']))

    def test_int_multi_coord(self):
        d = DataAssembly([1, 2, 3, 4, 5, 6],
                         coords={'a': ('multi_dim', [1, 1, 1, 1, 1, 1]),
                                 'b': ('multi_dim', ['a', 'a', 'a', 'b', 'b', 'b']),
                                 'c': ('multi_dim', ['a', 'b', 'c', 'd', 'e', 'f'])},
                         dims=['multi_dim'])
        g = d.multi_groupby(['a', 'b']).mean()
        assert g.equals(DataAssembly([2., 5.],
                                     coords={'a': ('multi_dim', [1, 1]), 'b': ('multi_dim', ['a', 'b'])},
                                     dims=['multi_dim']))

    def test_two_coord(self):
        assy = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
                         coords={'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                                 'down': ("a", [1, 1, 1, 1, 2, 2]),
                                 'sideways': ('b', ['x', 'y', 'z'])},
                         dims=['a', 'b'])
        assy_grouped = assy.multi_groupby(['up', 'down']).mean(dim="a")
        assy_2 = DataAssembly([[2.5, 3.5, 4.5], [8.5, 9.5, 10.5], [14.5, 15.5, 16.5]],
                                     coords={'up': ("a", ['alpha', 'beta', 'beta']),
                                             'down': ("a", [1, 1, 2]),
                                             'sideways': ('b', ['x', 'y', 'z'])},
                                     dims=['a', 'b'])
        assert assy_grouped.equals(assy_2)


class TestMultiDimApply:
    def test_unique_values(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)

    def test_unique_values_swappeddims(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['b', 'a'], lambda x, **_: x)
        assert g.equals(d)

    def test_nonindex_coord(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z'],
                                 # additional coordinate that has no index values.
                                 # This could e.g. be the result of `.sel(c='remnant')`
                                 'c': 'remnant'},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)  # also tests that `c` persists

    def test_subtract_mean(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'aa': ('a', ['a', 'a', 'b', 'b']),
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['aa', 'b'], lambda x, **_: x - x.mean())
        assert g.equals(DataAssembly([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
                                     coords={'a': ['a', 'b', 'c', 'd'],
                                             'aa': ('a', ['a', 'a', 'b', 'b']),
                                             'b': ['x', 'y', 'z']},
                                     dims=['a', 'b']))

    def test_multi_level(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
                         coords={'up': ("a", ['alpha', 'alpha', 'beta', 'beta', 'beta', 'beta']),
                                 'down': ("a", [1, 1, 1, 1, 2, 2]),
                                 'sideways': ('b', ['x', 'y', 'z'])},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)

