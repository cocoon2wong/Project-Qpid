"""
Base Dataset-Related Managers

It contains base classes to process dataset files.
All these objects are managed by the `AgentManager` object.

Pipelines:
- `BaseInputObjectManager` processes dataset files, and make a
    list of `BaseInputObject` objects to get ready to train;
- `BaseFilesManager` schedules all `BaseInputObjectManager` objects;
- `AgentManager` is the main manager to manage all above objects
    to make dataset files from all video clips from the dataset.
"""

from .__filesManager import BaseFilesManager
from .__inputManager import BaseInputManager
from .__inputObject import BaseInputObject
from .__inputObjectManager import BaseInputObjectManager
from .__picker import Annotation, AnnotationManager
