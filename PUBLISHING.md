# Publicar BioForge en PyPI — guía paso a paso

Esta guía es para **ti** (el autor). Publicar en PyPI hace que cualquiera pueda
`pip install bioforge`. Es público y el nombre se reclama, así que **ensayamos
primero en TestPyPI** (un PyPI de pruebas) antes del real.

> Estado actual: `pip install bioforge` dará el motor C completo en **Windows**
> y fallback NumPy (funcional, más lento) en Linux/Mac. Los wheels nativos de
> Linux/Mac vendrán en v2.4 (CI con cibuildwheel).

---

## 0. Una sola vez: instalar las herramientas

```bash
pip install build twine
```

## 1. Construir los paquetes (wheel + sdist)

Desde la raíz del repo:

```bash
python -m build
```

Esto crea en `dist/`:
- `bioforge-2.2.1-py3-none-any.whl`  (incluye el motor C de Windows)
- `bioforge-2.2.1.tar.gz`            (código fuente + engine.c para recompilar)

## 2. Ensayo en TestPyPI (recomendado)

1. Crea una cuenta en **https://test.pypi.org/account/register/**
2. Crea un token de API en **https://test.pypi.org/manage/account/token/**
   (ámbito: "Entire account"). Cópialo (empieza por `pypi-...`).
3. Sube:
   ```bash
   twine upload --repository testpypi dist/*
   ```
   Usuario: `__token__`  ·  Contraseña: el token que copiaste.
4. Prueba la instalación desde TestPyPI en un entorno limpio:
   ```bash
   python -m venv /tmp/probar && /tmp/probar/bin/pip install numpy
   /tmp/probar/bin/pip install -i https://test.pypi.org/simple/ bioforge
   /tmp/probar/bin/python -c "import bioforge; print(bioforge.__version__)"
   ```
   (En Windows: `Scripts\pip.exe` y `Scripts\python.exe`.)

## 3. Publicar en el PyPI real

Cuando el ensayo funcione:

1. Cuenta en **https://pypi.org/account/register/**
2. Token en **https://pypi.org/manage/account/token/**
3. Subir:
   ```bash
   twine upload dist/*
   ```
   Usuario: `__token__`  ·  Contraseña: el token de pypi.org.
4. ¡Listo! Ya se puede `pip install bioforge` en todo el mundo.

---

## Notas

- **El nombre `bioforge`**: compruébalo libre en https://pypi.org/project/bioforge/
  antes (si está ocupado, habría que elegir otro nombre en `pyproject.toml`).
- **No se puede re-subir la misma versión**: si te equivocas, sube `2.2.1`
  (cambia `__version__` en `bioforge/__init__.py`).
- **Seguridad**: nunca subas el token a git. Úsalo solo al ejecutar `twine`.
- **v2.4 (futuro)**: wheels nativos por plataforma con cibuildwheel + GitHub
  Actions, para que Linux/Mac también tengan el motor C rápido al instalar.
