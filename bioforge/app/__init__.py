"""
bioforge.app — la app de escritorio de BioForge ("la otra cara del motor").

Una ventana NATIVA y LOCAL (PyWebview) sobre el mismo motor ``bioforge``, pensada
para no-programadores: analizar ADN a clics, sin escribir código y sin que los datos
salgan de la máquina (ADN Edge). Se distribuye de dos formas, ambas a la misma versión:

  · como ejecutable ``.exe`` autocontenido (doble clic, sin instalar Python), y
  · desde el paquete:  ``pip install bioforge[app]``  y luego el comando  ``bioforge-app``.

Requiere el extra ``app`` (pywebview + h5py). El backend (``bioforge.app.backend.Api``)
es Python puro y se prueba SIN abrir ninguna ventana.
"""
