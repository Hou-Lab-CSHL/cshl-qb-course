import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from contextlib import contextmanager

def read_3d_data(data_dir):
    """Load 3D landmark data from Anipose.

    ### Arguments
    - `data_dir`: the Anipose directory for the video to load

    ### Returns
    A dictionary of 3D data where each key is a landmark name
    and the associated value is an array of shape `(time, 3)`
    """
    files = glob(os.sep.join([data_dir, "pose-3d", "*.csv"]))
    data = pd.read_csv(files[0])
    cols = data.head() # name of all the columns
    landmark_names = np.unique([s.split('_')[0]
                                for s in cols if s.endswith(('_x', '_y', '_z'))])
    landmarks = {landmark: np.stack([data[f"{landmark}_x"],
                                     data[f"{landmark}_y"],
                                     data[f"{landmark}_z"]], axis=-1)
                 for landmark in landmark_names}

    return landmarks

class VideoFrames:
    """
    A (potentially time shifted) video indexed by frames for a recording session
    cropped to a bounding box region.

    Arguments:
    - `path`: a path to the video file
    - `shift = 0`: the +/- shift in units of video frames
    - `bounds = [None, None, None, None]`: a tuple of the form
        `[xstart, xend, ystart, yend]` (set any element to `None` to use the
        max bounds)
    """
    def __init__(self, path, shift = 0, bounds = [None, None, None, None]):
        self.shift = shift
        self.bounds = bounds
        self.path = path

    @contextmanager
    def opencv_capture(self):
        cap = cv2.VideoCapture(self.path)
        try:
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def __iter__(self):
        # use opencv for faster iteration
        sx, ex, sy, ey = self.bounds
        with self.opencv_capture() as video:
            nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.set(cv2.CAP_PROP_POS_FRAMES, max(self.shift, 0))
            for _ in range(nframes):
                ret, frame = video.read()
                if ret:
                    yield frame[sy:ey, sx:ex]
                else:
                    break

    def __str__(self):
        return str(self.path)

def sns_setup(context = "paper", palette = "Set2", font = "Arial"):
    """Set up Seaborn to have the default lab theme.
    You can override the default plotting context and color palette,
    but we recommend calling this function with no arguments.

    ### Arguments
    - `context`: the plotting context (see `seaborn.set_theme`)
    - `palette`: a valid Seaborn color palette (see `seaborn.set_theme`)
    """
    if context == "paper":
        # Taken from
        # https://github.com/cxrodgers/my/blob/4e3448251824629f1ce7031a8db045add39ca535/plot.py#L561
        rcparams = {
            # For PDF imports:
            # Not sure what this does
            "ps.useafm": True,
            # Makes it so that the text is editable
            "pdf.fonttype": 42,
            # seems to work better
            "svg.fonttype": "none",
            # adjust font sizes
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11
        }
        sns.set_theme(context="paper",
                      font=font,
                      font_scale=1,
                      style="ticks",
                      palette=palette,
                      rc=rcparams)
    else:
        sns.set_theme(context=context,
                      font=font,
                      font_scale=1.125,
                      style="ticks",
                      palette=palette)
