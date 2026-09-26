"""Scene base class shared by all scene modules."""
from core import blank


class Scene:
    dur = 5.0
    # post: grain amount, vignette strength, letterbox ratio (0 = none), chroma shift px
    grain = 0.045
    vig = 0.45
    lbox = 0.0
    ca = 0.0
    fade_in = 0.0
    fade_out = 0.0
    flash_in = 0.0      # seconds of white flash decay at scene start
    xfade = 0.0         # crossfade from previous scene (seconds)

    def __init__(self):
        self._ready = False

    def ensure(self):
        if not self._ready:
            self.setup()
            self._ready = True

    def setup(self):
        pass

    def render(self, t, f):
        return blank()

    def overlay(self, img, t, f):
        return img
