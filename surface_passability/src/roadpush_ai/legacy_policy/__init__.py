"""Stable bindings for the frozen pre-package policy modules."""

from . import integrated_sidewalk as integrated
from . import grounded_footprint as grounded
from . import rowwise_completion as a2
from . import vehicle_geometry as geometry

outer = integrated.outer
base = outer.base
policy = integrated.policy
curb_probe = integrated.curb_probe
