import numpy as np

from oap_supcon.experiment import retrieval_metrics


def test_retrieval_protocol_excludes_forbidden_gallery_pairs():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_labels = np.array([0, 1])
    probe = np.array([[1.0, 0.0]], dtype=np.float32)
    probe_labels = np.array([0])
    unrestricted, _ = retrieval_metrics(gallery, gallery_labels, probe, probe_labels)
    restricted, _ = retrieval_metrics(
        gallery,
        gallery_labels,
        probe,
        probe_labels,
        exclusion_mask=np.array([[True, False]]),
    )
    assert unrestricted["rank1"] == 1.0
    assert restricted["rank1"] == 0.0
