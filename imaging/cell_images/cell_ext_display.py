"""Four-channel review rendering following cell_ext_err2's display convention."""
import os
from pathlib import Path
import ast
import cv2
import numpy as np

REFERENCE=Path(os.environ.get('CELL_EXT_REFERENCE', 'cell_ext_err2.py'))
UTILS=REFERENCE.parent/'utils.py'


def preprocess_frame(raw,gains):
    """Gain exactly once, saturate before integer conversion, full-frame top-hat."""
    if raw.dtype!=np.uint16 or raw.ndim!=3 or raw.shape[-1]!=4:raise ValueError('Expected uint16 DAPI/TRITC/CY5/FITC frame')
    gains=np.asarray(gains,dtype=np.float32)
    if gains.shape!=(4,) or not np.isfinite(gains).all() or (gains<=0).any():raise ValueError('Invalid explicit gains')
    gained=np.clip(raw.astype(np.float32)*gains,0,65535).astype(np.uint16)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(75,75))
    return cv2.morphologyEx(gained,cv2.MORPH_TOPHAT,kernel)


def rgb(raw):
    """Exact to_rgb: FITC contributes white to red, green AND blue."""
    if raw.dtype!=np.uint16 or raw.shape[-1]!=4:raise ValueError('Expected four uint16 channels')
    normalized=raw.astype(np.float32)/65535.0
    color=np.concatenate([normalized[...,1:2]+normalized[...,3:4],normalized[...,2:3]+normalized[...,3:4],normalized[...,0:1]+normalized[...,3:4]],axis=-1)
    color[color>1]=1
    return (color*255).astype(np.uint8)


def channel_u8(raw):
    return (raw.astype(np.float32)/65535.0*255).astype(np.uint8)


def reference_oracle():
    """Read/compile only pure requested methods; do not import/run upstream CLI."""
    tree=ast.parse(REFERENCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CellExtractor')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['to_rgb','preprocess_frame']]
    util=ast.parse(UTILS.read_text());top=next(n for n in util.body if isinstance(n,ast.FunctionDef) and n.name=='apply_top_hat')
    namespace={'np':np,'cv2':cv2};exec(compile(ast.Module(body=[top,*methods],type_ignores=[]),str(REFERENCE),'exec'),namespace)
    return namespace
