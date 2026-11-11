"""Files the sandbox puts on PATH and PYTHONPATH (see halflife.sandbox).

``sitecustomize.py`` logs Python socket use; the executable shims (``curl``,
``wget``, ``pip``, ``pip3``, ``npm``, ``git``) log and refuse network use.
Importing this package has no side effects; the logger installs itself only
when Python imports it as the top-level ``sitecustomize`` module.
"""
