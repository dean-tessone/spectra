"""Outcome-independent selection of the earliest draw, tube and slide numbers."""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from .patient_identity import patient_identity_key

POLICY_ID = 'earliest_draw_tube_slides_v1'


def slide_sequence(slide_id: str) -> int:
    """CSI tube ID is the first five characters; the remaining digits are slides."""
    if not isinstance(slide_id, str) or not re.fullmatch(r'[A-Za-z0-9]{5}[0-9]+', slide_id):
        raise ValueError(f'Invalid tube-prefixed slide ID: {slide_id!r}')
    return int(slide_id[5:])


def select_earliest_slides(frame: pd.DataFrame, *, max_slides: int = 2) -> pd.DataFrame:
    """Select earliest timestamp, lowest tube ID, then lowest numeric slide suffix.

    Input rows must already pass acquisition, clinical and sample-type eligibility.
    Select within a single tube. Preserve original group IDs as provenance and
    create new deterministic bags, permitting the earliest two eligible slides
    to span former groups. Slide numbers establish ordering, not observed times.
    """
    required = {'donor_id','patient_id','slide_id','group_id','draw_date','label','study'}
    if missing := required - set(frame):
        raise ValueError(f'Missing selection columns: {sorted(missing)}')
    if frame.empty or max_slides < 1 or frame.slide_id.duplicated().any():
        raise ValueError('Nonempty unique slides and a positive bag size are required')
    f = frame.copy()
    f['original_donor_id'] = f.get('original_donor_id', f.donor_id)
    f['donor_id'] = f.donor_id.map(patient_identity_key)
    f['_draw'] = pd.to_datetime(f.draw_date, errors='raise')
    if f._draw.isna().any():
        raise ValueError('Missing draw timestamp')
    f['_tube'] = f.slide_id.str[:5].str.upper()
    f['_sequence'] = f.slide_id.map(slide_sequence)
    if 'tube_id' in f and not f.tube_id.str.upper().eq(f._tube).all():
        raise ValueError('Tube metadata disagrees with slide-ID prefix')
    if f.groupby(['donor_id','_draw','_tube','_sequence']).size().gt(1).any():
        raise ValueError('Ambiguous slide sequence within a tube')
    output = []
    for donor, rows in f.groupby('donor_id', sort=True):
        if rows.label.astype(int).nunique() != 1 or rows.study.nunique() != 1:
            raise ValueError(f'Conflicting label/study for canonical donor {donor}')
        if 'split' in rows and rows.split.nunique() != 1:
            raise ValueError(f'Canonical donor crosses historical splits: {donor}')
        first_draw = rows._draw.min()
        first = rows[rows._draw.eq(first_draw)]
        tube = first._tube.min()
        selected = first[first._tube.eq(tube)].sort_values('_sequence').head(max_slides).copy()
        selected['original_group_id'] = selected.group_id
        digest = hashlib.sha256(':'.join([POLICY_ID, donor, str(first_draw), tube,
                                         *selected.slide_id]).encode()).hexdigest()[:12]
        selected['group_id'] = f'earliest__{donor}__{tube}__{digest}'
        selected['tube_id'] = tube
        selected['selection_policy'] = POLICY_ID
        if 'split' in selected:
            selected['historical_split'] = selected.split
        output.append(selected)
    return pd.concat(output, ignore_index=True).drop(columns=['_draw','_tube','_sequence'])
