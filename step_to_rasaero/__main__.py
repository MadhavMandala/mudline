"""Allow ``python -m step_to_rasaero``.

``main.py`` already had an argparse entry point, but without this file the
package could only be invoked by path, which meant the documented command did
not work.
"""

from .main import main

raise SystemExit(main())
