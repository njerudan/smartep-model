import numpy as np
import pandas as pd
import pytest

from smartep.config import deep_merge, parse_override
from smartep.data.splits import frame_hash, grouped_split
from smartep.features.sequence import SequenceNormalizer, build_weekly_tensor
from smartep.features.tabular import TabularFeaturizer, build_legacy_tabular
from smartep.features.text import TextEncoder
from smartep.features.watch_history import build_watch_history
from smartep.synth.legacy import LEGACY_TOPICS, generate_legacy_tables, label_matrix


@pytest.fixture(scope="module")
def tables():
    return generate_legacy_tables(120, seed=1)


def test_legacy_tables_are_deterministic(tables):
    again = generate_legacy_tables(120, seed=1)
    assert frame_hash(tables.surveys, tables.skill_gaps) == frame_hash(again.surveys, again.skill_gaps)
    labels = label_matrix(tables)
    assert labels.shape == (120, len(LEGACY_TOPICS)) and set(np.unique(labels)) <= {0, 1}


def test_grouped_split_keeps_groups_together():
    groups = np.repeat([f"E{i}" for i in range(50)], 3)
    split = grouped_split(groups, 0.2, 0.2, seed=0)
    parts = [set(groups[idx]) for idx in (split.train, split.val, split.test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert sum(split.sizes().values()) == len(groups)
    assert len(parts[2]) == 10 and len(parts[1]) == 10


def test_featurizer_uses_training_rows_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "dept": ["A", "B", "A"]})
    test = pd.DataFrame({"x": [100.0], "dept": ["C"]})  # unseen level
    f = TabularFeaturizer(categorical=["dept"]).fit(train)
    assert f.feature_names == ["x", "dept=A", "dept=B"]
    np.testing.assert_allclose(f.transform(train).mean(axis=0), 0, atol=1e-6)
    out = f.transform(test)
    assert out[0, 0] > 10  # scaled with training statistics, not refitted
    np.testing.assert_allclose(out[0, 1:], f.scaler.transform([[100.0, 0.0, 0.0]])[0, 1:], atol=1e-6)


def test_legacy_tabular_has_one_row_per_educator(tables):
    tab = build_legacy_tabular(tables)
    assert len(tab) == 120 and tab.drop(columns="department").isna().sum().sum() == 0


def test_sequence_tensor_and_normalizer(tables):
    seq = build_weekly_tensor(tables.lms_logs, tables.educator_ids, 5)
    assert seq.shape == (120, 5, 3)
    norm = SequenceNormalizer().fit(seq[:80])
    np.testing.assert_allclose(norm.transform(seq[:80]).reshape(-1, 3).mean(axis=0), 0, atol=1e-4)


def test_watch_history_mask_matches_rows(tables):
    feats, mask = build_watch_history(tables.video_watch_logs, tables.video_catalog, tables.educator_ids,
                                      LEGACY_TOPICS, max_neighbors=15)
    counts = tables.video_watch_logs.groupby("educator_id").size().reindex(tables.educator_ids).clip(upper=15)
    np.testing.assert_array_equal(mask.sum(axis=1), counts.to_numpy())
    assert np.all(feats[..., :-1].sum(axis=-1)[mask == 1] == 1)


def test_text_encoder_dims():
    texts = ["lecture on AI for diploma", "lecture on cloud for postgraduate", "databases intro", "networks"]
    enc = TextEncoder("tfidf_svd", dim=2, seed=0).fit(texts)
    assert enc.transform(texts).shape == (4, 2)
    assert TextEncoder("none").fit(texts).transform(texts).shape == (4, 0)


def test_config_overrides():
    assert parse_override("data.n_educators=300") == {"data": {"n_educators": 300}}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}}
