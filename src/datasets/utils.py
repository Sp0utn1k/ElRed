import numpy as np
from typing import List, Tuple

from .latlon_data import LatLonData

def merge(subsets: List[LatLonData]) -> LatLonData:
    """
    Merge a list of LatLonData instances into a single LatLonData instance.
    Returns:
        merged_instance: LatLonData
    """
    labels = None
    for subset in subsets:
        if subset.labels is not None:
            labels = []
            break
    latlons = []
    ellipses = []
    idx = []
    timestamps = []
    max_label = 0
    for i, subset in enumerate(subsets):
        latlons.append(subset.latlons)
        ellipses.append(subset.ellipses)
        idx.append(subset.idx)
        if subset.timestamps is not None:
            timestamps.append(subset.timestamps)
        if labels is not None:
            new_labels = subset.labels if subset.labels is not None else np.zeros(len(subset), dtype=int)
            new_labels += max_label + 1
            labels.append(new_labels)
            max_label = new_labels.max()
    latlons = np.concatenate(latlons, axis=0)
    ellipses = np.concatenate(ellipses, axis=0)
    idx = np.concatenate(idx, axis=0)
    if labels is not None:
        labels = np.concatenate(labels, axis=0)
    # Verify all indices are unique
    assert len(np.unique(idx)) == len(idx), "Merged indices are not unique!"
    # Handle timestamps: only merge if all have timestamps, else None
    if all(inst.timestamps is not None for inst in subsets):
        timestamps = np.concatenate(timestamps, axis=0)
    else:
        timestamps = None
    # Merge metadata: just keep a list of all
    metadata = {'sources': [inst.metadata for inst in subsets]}
    merged = LatLonData(
        latlons=latlons,
        ellipses=ellipses,
        idx=idx,
        timestamps=timestamps,
        metadata=metadata,
        labels=labels
    )
    return merged
