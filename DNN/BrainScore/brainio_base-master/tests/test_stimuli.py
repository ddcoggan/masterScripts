import pandas as pd

from brainio_base.stimuli import StimulusSet


class TestPreservation:
    def test_subselection(self):
        stimulus_set = StimulusSet([{'image_id': i} for i in range(100)])
        stimulus_set.image_paths = {i: f'/dummy/path/{i}' for i in range(100)}
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(stimulus_set['image_id'].values[:3])]
        assert stimulus_set.get_image(0) is not None

    def test_pd_concat(self):
        s1 = StimulusSet([{'image_id': i} for i in range(10)])
        s1.image_paths = {i: f'/dummy/path/{i}' for i in range(10)}
        s2 = StimulusSet([{'image_id': i} for i in range(10, 20)])
        s2.image_paths = {i: f'/dummy/path/{i}' for i in range(10, 20)}
        s = pd.concat((s1, s2))
        s.image_paths = {**s1.image_paths, **s2.image_paths}
        assert s.get_image(1) is not None
        assert s.get_image(11) is not None
