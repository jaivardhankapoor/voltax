"""Import-time configuration (imported first by ``voltax/__init__.py``).

Voltax enables 64-bit floats: circuit equations mix femtofarads, kilohms and
picoamps, and single precision is not enough for reliable Newton convergence
or accurate gradients.
"""

import jax

jax.config.update("jax_enable_x64", True)
