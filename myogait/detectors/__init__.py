# Detector sub-package.  Importing sub-modules here triggers their
# register_event_method() calls so callers only need:
#
#   from myogait.detectors import learned_contact_detector  # noqa: F401
#
# or rely on the lazy import in the top-level myogait/__init__.py.
