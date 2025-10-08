import numpy as np
from typing import Optional, Iterator, Union, List
import warnings

class LatLonData:
    def __init__(self, latlons: np.ndarray, ellipses: np.ndarray, 
                 idx: Optional[np.ndarray] = None, 
                 timestamps: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None,
                 bbox_multiplier: float = 1.5,
                 labels: Optional[np.ndarray] = None,
                 parent = None):
        assert len(latlons) == len(ellipses), "latlons and ellipses must have same length"
        self.latlons = latlons
        self.ellipses = ellipses
        self.idx = idx if idx is not None else np.arange(len(latlons))
        self.timestamps = timestamps  # For temporal analysis
        self.metadata = metadata or {}
        self._cache = {}  # initialize cache
        self.bbox_multiplier = bbox_multiplier
        # Add labels vector, defaulting to zeros
        self.labels = labels
        # Validate all arrays have consistent length
        arrays_to_check = [self.latlons, self.ellipses, self.idx]
        if self.timestamps is not None:
            arrays_to_check.append(self.timestamps)
        lengths = [len(arr) for arr in arrays_to_check]
        assert all(l == lengths[0] for l in lengths), "All arrays must have same length"
        
        self.parent = parent
    
    def __len__(self) -> int:
        return len(self.latlons)
    
    def __getitem__(self, key: Union[slice, np.ndarray, List[int]]) -> 'LatLonData':
        """Enable indexing/slicing: data[mask] or data[0:10]"""
        obj = LatLonData(
            latlons=self.latlons[key],
            ellipses=self.ellipses[key],
            idx=self.idx[key],
            timestamps=self.timestamps[key] if self.timestamps is not None else None,
            metadata=self.metadata.copy(),
            bbox_multiplier=self.bbox_multiplier,
            labels=self.labels[key] if self.labels is not None else None,
            parent = self.parent or self
        )
        # transfer cached arrays to subset
        obj._cache = {name: arr[key] for name, arr in self._cache.items()}
        return obj
    
    def get_by_index(self, indices: np.ndarray) -> 'LatLonData':
        """Get subset by original indices."""
        mask = np.isin(self.idx, indices)
        return self[mask]
    
    def split(self, subsets: List[np.ndarray]) -> Iterator['LatLonData']:
        """Split into multiple LatLonData objects"""
        for subset in subsets:
            yield self[subset]
    
    @property
    def is_empty(self) -> bool:
        return len(self) == 0
    
    def copy(self) -> 'LatLonData':
        """Create a deep copy"""
        obj = LatLonData(
            latlons=self.latlons.copy(),
            ellipses=self.ellipses.copy(),
            idx=self.idx.copy(),
            timestamps=self.timestamps.copy() if self.timestamps is not None else None,
            metadata=self.metadata.copy(),
            bbox_multiplier=self.bbox_multiplier,
            labels=self.labels.copy()
        )
        # preserve any already-computed cache entries
        obj._cache = self._cache.copy()
        return obj
    
    def get_bounds(self) -> dict:
        """Get lat/lon bounds - useful for projection decisions"""
        return {
            'lat_min': self.lats.min(),
            'lat_max': self.lats.max(),
            'lon_min': self.lons.min(),
            'lon_max': self.lons.max()
        }
        
    def df_subset(self, df):
        return df.iloc[self.idx]
    
    @property
    def lats(self) -> np.ndarray:
        """Return array of latitudes."""
        return self.latlons[:, 0]

    @property
    def lons(self) -> np.ndarray:
        """Return array of longitudes."""
        return self.latlons[:, 1]

    @property
    def majors(self) -> np.ndarray:
        """Return array of ellipse major axes."""
        return self.ellipses[:, 0]

    @property
    def minors(self) -> np.ndarray:
        """Return array of ellipse minor axes."""
        return self.ellipses[:, 1]

    @property
    def bearings(self) -> np.ndarray:
        """Return array of ellipse bearings (angles)."""
        return self.ellipses[:, 2]

    def clear_cache(self) -> None:
        """Clear all cached projections."""
        self._cache.clear()

    def clear_cache_item(self, name: str) -> None:
        """Clear a specific cached entry (e.g. 'proj_ns' or 'proj_ew')."""
        self._cache.pop(name, None)

    @property
    def proj_ns(self) -> np.ndarray:
        """Projected ellipse lengths along North-South axis."""
        if 'proj_ns' not in self._cache:
            self._cache['proj_ns'] = self._compute_proj_ns()
        return self._cache['proj_ns']

    @property
    def proj_ew(self) -> np.ndarray:
        """Projected ellipse lengths along East-West axis."""
        if 'proj_ew' not in self._cache:
            self._cache['proj_ew'] = self._compute_proj_ew()
        return self._cache['proj_ew']

    def _compute_proj_ns(self) -> np.ndarray:
        """Correct NS half-extent (support function) for a rotated ellipse."""
        a = self.majors  # 95% semi-major in meters
        b = self.minors  # 95% semi-minor in meters
        th = np.radians(self.bearings)  # bearings: deg, clockwise from North
        # NS component: a*cosθ, b*sinθ
        return np.sqrt((a * np.cos(th))**2 + (b * np.sin(th))**2)

    def _compute_proj_ew(self) -> np.ndarray:
        """Correct EW half-extent (support function) for a rotated ellipse."""
        a = self.majors
        b = self.minors
        th = np.radians(self.bearings)
        # EW component: a*sinθ, b*cosθ
        return np.sqrt((a * np.sin(th))**2 + (b * np.cos(th))**2)

    @property
    def bbox_min(self) -> np.ndarray:
        """Return array of minimum [lat, lon] for each point’s bounding box."""
        cache_key = f'bbox_min_{self.bbox_multiplier}'
        if cache_key not in self._cache:
            self._cache[cache_key] = self._compute_bbox_min()
        return self._cache[cache_key]

    @property
    def bbox_max(self) -> np.ndarray:
        """Return array of maximum [lat, lon] for each point’s bounding box."""
        cache_key = f'bbox_max_{self.bbox_multiplier}'
        if cache_key not in self._cache:
            self._cache[cache_key] = self._compute_bbox_max()
        return self._cache[cache_key]

    def _meters_to_degree_offsets(self, ns, ew, lat):
        meters_per_deg_lat = 111320.0
        deg_lat = ns / meters_per_deg_lat
        # avoid division by ~0 at high latitudes
        coslat = np.cos(np.radians(lat))
        coslat = np.clip(coslat, 1e-3, None)
        meters_per_deg_lon = meters_per_deg_lat * coslat
        deg_lon = ew / meters_per_deg_lon
        return deg_lat, deg_lon


    def _compute_bbox_min(self) -> np.ndarray:
        lat = self.lats
        lon = self.lons
        ns = self.proj_ns * self.bbox_multiplier
        ew = self.proj_ew * self.bbox_multiplier
        deg_lat, deg_lon = self._meters_to_degree_offsets(ns, ew, lat)
        lat_min = lat - deg_lat
        lon_min = lon - deg_lon
        # If bbox crosses a pole, set longitude to cover full range
        crosses_pole = (lat_min < -90) | ((lat + deg_lat) > 90)
        lon_min = np.where(crosses_pole, -180.0, lon_min)
        lat_min = np.clip(lat_min, -90, 90)
        # Raise warning if longitude crosses the -180°/180° meridian
        if np.any(np.abs(lon_min) > 180):
            warnings.warn("Bounding box longitude minimum crosses the -180°/180° meridian.")
        return np.column_stack((lat_min, lon_min))

    def _compute_bbox_max(self) -> np.ndarray:
        lat = self.lats
        lon = self.lons
        ns = self.proj_ns * self.bbox_multiplier
        ew = self.proj_ew * self.bbox_multiplier
        deg_lat, deg_lon = self._meters_to_degree_offsets(ns, ew, lat)
        lat_max = lat + deg_lat
        lon_max = lon + deg_lon
        # If bbox crosses a pole, set longitude to cover full range
        crosses_pole = ((lat - deg_lat) < -90) | (lat_max > 90)
        lon_max = np.where(crosses_pole, 180.0, lon_max)
        lat_max = np.clip(lat_max, -90, 90)
        # Raise warning if longitude crosses the -180°/180° meridian
        if np.any(np.abs(lon_max) > 180):
            warnings.warn("Bounding box longitude maximum crosses the -180°/180° meridian.")
        return np.column_stack((lat_max, lon_max))
