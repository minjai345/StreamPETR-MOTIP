"""MOTIP modules ported from SparseBEV.

Drop-in copy with two cleanups:
- tracker.py → id_decoder.py (file name now matches the contained class)
- augmentation.py removed (it was dead code; SparseBEV's compute_id_loss
  inlines its own occlusion/switch augmentation and never imported the
  augment_trajectories function).
"""
from .id_dictionary import IDDictionary
from .pos_encoding import Positional3DEncoding
from .tracklet import TrackletFormer
from .id_decoder import IDDecoder

__all__ = ['IDDictionary', 'Positional3DEncoding', 'TrackletFormer', 'IDDecoder']
