# 🗿 modelo3d — De foto a modelo 3D imprimible

Convertí una foto (o tres vistas del mismo objeto) en un archivo **STL listo para imprimir**, sin saber nada de programación.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<OWNER>/<REPO>/blob/main/modelo3d.ipynb)

## Cómo usarlo

1. **Abrí el notebook en Colab** haciendo clic en el botón de arriba.
2. **Activá la GPU:** andá a *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar*.
3. **Ejecutá la celda de instalación** y esperá el mensaje ✅.
4. **Usá la app** que aparece al final de la página.

> ⚠️ Cuando la sesión de Colab se cierre, los archivos se borran. Descargá tu STL apenas lo generes.

## Consejos para la foto

- Buena luz, sin flash directo
- Fondo liso y de color parejo
- Un solo objeto, centrado y completo
- Sin manos sosteniendo el objeto
- En modo varias fotos: misma distancia y altura en las tres tomas

## Para desarrolladores

### Requisitos

- Python 3.10+
- Google Colab con T4 GPU (para ejecución completa)

### Dependencias de desarrollo

```bash
pip install pytest nbformat trimesh pymeshfix manifold3d pillow numpy ruff gradio
```

### Ejecutar tests

```bash
python -m pytest tests/ -v
```

### Estructura del notebook

| Celda | Tags | Contenido |
|-------|------|-----------|
| 0 | — | Introducción (markdown) |
| 1 | `setup` | GPU gate, instalación, descarga de modelos |
| 2 | `core` | Configuración y presets de tamaño |
| 3 | `core` | Validación de fotos y catálogo de errores |
| 4 | `core` | Núcleo geométrico: reparar, escalar, base, exportar |
| 5 | `core` | Selección de motor y reintento por OOM |
| 6 | `app`, `core` | Aplicación Gradio |
| 7 | `selftest`, `core` | Modo prueba con imagen de ejemplo |

### Protocolo manual Colab T4

Antes de entregar el notebook a un usuario, verificar los 5 casos:

- [ ] **Objeto simple** (una foto): foto de un objeto con fondo liso → STL correcto
- [ ] **Mascota** (una foto): foto de un animal → STL correcto
- [ ] **Busto** (una foto): busto o figura humana → STL correcto
- [ ] **Multiview happy path** (tres fotos): frente + perfil + espalda → STL correcto usando motor multiview
- [ ] **Fallback a una foto** (dos fotos): solo frente + perfil → aviso explícito, usa motor de una foto

Registrar resultados (pass/fail + notas) antes de entregar.

## Licencia

MIT — ver [LICENSE](LICENSE).
