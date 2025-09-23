import numpy as np
from typing import List, Tuple

from .latlon_data import LatLonData

def merge(instances: List[LatLonData]) -> LatLonData:
    """
    Merge a list of LatLonData instances into a single LatLonData instance.
    Returns:
        merged_instance: LatLonData
    """
    latlons = []
    ellipses = []
    idx = []
    timestamps = []
    labels = []
    max_label = 0
    for i, inst in enumerate(instances):
        latlons.append(inst.latlons)
        ellipses.append(inst.ellipses)
        idx.append(inst.idx)
        if inst.timestamps is not None:
            timestamps.append(inst.timestamps)
        new_labels = inst.labels + max_label + 1
        labels.append(new_labels)
        max_label = new_labels.max()
    latlons = np.concatenate(latlons, axis=0)
    ellipses = np.concatenate(ellipses, axis=0)
    idx = np.concatenate(idx, axis=0)
    labels = np.concatenate(labels, axis=0)
    # Verify all indices are unique
    assert len(np.unique(idx)) == len(idx), "Merged indices are not unique!"
    # Handle timestamps: only merge if all have timestamps, else None
    if all(inst.timestamps is not None for inst in instances):
        timestamps = np.concatenate(timestamps, axis=0)
    else:
        timestamps = None
    # Merge metadata: just keep a list of all
    metadata = {'sources': [inst.metadata for inst in instances]}
    merged = LatLonData(
        latlons=latlons,
        ellipses=ellipses,
        idx=idx,
        timestamps=timestamps,
        metadata=metadata,
        labels=labels
    )
    return merged
