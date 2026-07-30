

![Línea de modelos corporales](assets/readme/body-model-lineup.png)

# body-models

`body-models` proporciona una interfaz compartida para modelos paramétricos de cuerpo humano, cabeza, mano, anatómicos, de medidas y robots en NumPy, PyTorch y JAX.

Documentación: https://abcamiletto.github.io/body-models/

## Características

- API compartida para modelos de cuerpo humano, anatómicos, manos, cabezas, medidas y robots
- Backends para NumPy, PyTorch y JAX
- Propagaciones hacia adelante separadas para malla y esqueleto con `forward_vertices()` y `forward_skeleton()`
- Identidades precalculadas para posturas repetidas con parámetros de forma/expresión fijos
- Simplificación de malla y propagaciones hacia adelante para subconjuntos de vértices en modelos de malla compatibles
- Múltiples representaciones de rotación para modelos de postura compatibles

## Instalación

```bash
uv add body-models
```

Instala los extras opcionales cuando sea necesario:

```bash
uv add "body-models[torch]"
uv add "body-models[jax]"
```

## Primeros Pasos

```python
import body_models

model = body_models.create_model("smpl", backend="torch")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Descubre los nombres de modelos disponibles con `body_models.list_models()`. Las opciones del modelo, como `gender="male"` o `side="left"`, se pasan como argumentos de palabras clave (kwargs) del constructor.

Cuando los parámetros de identidad dependientes de la forma permanecen fijos en múltiples posturas, prepáralos una vez y pasa el diccionario devuelto a través de `identity`. Esto evita recomputar las articulaciones en reposo, los desplazamientos locales y los vértices en reposo en cada propagación hacia adelante.

```python
shape = params.pop("shape")
identity = model.prepare_identity(shape)

vertices = model.forward_vertices(**params, identity=identity)
skeleton = model.forward_skeleton(**params, identity=identity)
```

Para modelos con un estado en reposo dependiente de la expresión, como SMPL-X y FLAME, pasa ambos controles de identidad a `prepare_identity(shape, expression)`. El trabajo exclusivo de esqueletos puede usar `skip_vertices=True` para evitar preparar los vértices en reposo.

## Modelos Soportados

- Cuerpo completo: SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA, GarmentMeasurements
- Anatómicos: SKEL, MyoFullBody
- Cabezas: FLAME
- Manos: MANO
- Robots: BrainCo, G1, SmplHumanoid

Consulta la [documentación de modelos](https://abcamiletto.github.io/body-models/#supported-models) para la configuración, backends compatibles, entradas y comportamiento específico de cada modelo.

## Desarrollo

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
```

## Licencia

Consulta la documentación y los proyectos originales de los modelos para los términos de licencia específicos de cada modelo.
