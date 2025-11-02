# 🐳 Guía Paso a Paso: Docker (Código + Dependencias)

Esta guía te muestra cómo ejecutar el proyecto ML ClashRoyale usando Docker, **sin incluir los datos grandes** (~10GB) en la imagen.

## ✅ Verificación previa

### 1. Verificar que Docker Desktop está instalado y corriendo

Abre PowerShell o CMD y ejecuta:
```bash
docker --version
docker-compose --version
```

**Si no tienes Docker Desktop:**
- Descarga desde: https://www.docker.com/products/docker-desktop/
- Instala y reinicia tu PC
- Asegúrate de que Docker Desktop está **corriendo** (verás el icono de la ballena en la barra de tareas)

## 🚀 Pasos de ejecución

### Paso 1: Abrir terminal en el proyecto

```bash
cd C:\Users\Usuario\Documents\GitHub\ML_ClashRoyale
```

### Paso 2: Construir la imagen Docker

**Esto incluye:**
- ✅ Python 3.11
- ✅ Airflow 3.1.0
- ✅ Kedro y dependencias ML (scikit-learn, xgboost, lightgbm, etc.)
- ✅ Todo el código del proyecto (`src/`)
- ❌ **NO incluye datos** (se montan como volumen)

```bash
docker-compose build
```

**Tiempo estimado:** 5-15 minutos  
**Tamaño de imagen:** ~1-2 GB (vs ~12 GB si incluyera los datos)

**Lo que verás:**
- Descarga de imagen base de Python
- Instalación de dependencias del sistema
- Instalación de Python packages (Airflow, Kedro, etc.)
- Copia del código del proyecto

### Paso 3: Iniciar Airflow

```bash
docker-compose up -d
```

El flag `-d` ejecuta el contenedor en segundo plano (detached mode).

**Lo que hace:**
- Crea el contenedor `clashroyale_ml_airflow`
- Inicializa la base de datos de Airflow (si no existe)
- Inicia webserver y scheduler en modo standalone
- Monta los datos desde `./data` como volumen

### Paso 4: Verificar que está corriendo

```bash
# Ver estado de los contenedores
docker-compose ps

# Deberías ver algo como:
# NAME                    STATUS          PORTS
# clashroyale_ml_airflow  Up X minutes    0.0.0.0:8080->8080/tcp
```

### Paso 5: Ver logs (opcional)

```bash
# Ver logs en tiempo real
docker-compose logs -f airflow

# Ver solo los últimos 100 logs
docker-compose logs --tail=100 airflow
```

**Espera a ver:**
```
airflow-webserver | INFO: 127.0.0.1:XXXXX - "GET /health HTTP/1.1" 200
```
Esto significa que Airflow está listo.

### Paso 6: Acceder a la interfaz web

1. Abre tu navegador
2. Ve a: **http://localhost:8080**
3. Inicia sesión con:
   - **Usuario**: `admin`
   - **Contraseña**: `admin`

### Paso 7: Verificar DAGs

En la interfaz de Airflow deberías ver 5 DAGs:
- ✅ `clashroyale_ml_with_datacleaning` (completo)
- ✅ `clashroyale_ml_no_datacleaning` (sin data cleaning)
- ✅ `classification_ml_pipeline` (solo clasificación)
- ✅ `regression_ml_pipeline` (solo regresión)
- ✅ `data_cleaning_only` (solo data cleaning)

## 🛑 Comandos útiles

### Detener Airflow
```bash
docker-compose down
```

### Detener y eliminar volúmenes (reset completo)
```bash
docker-compose down -v
```
⚠️ **Cuidado:** Esto elimina la base de datos de Airflow y tendrás que iniciar sesión de nuevo.

### Reiniciar Airflow
```bash
docker-compose up -d
```

### Reconstruir imagen (después de cambios en Dockerfile o requirements.txt)
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Ejecutar comandos dentro del contenedor
```bash
# Abrir shell dentro del contenedor
docker-compose exec airflow bash

# Ejecutar un comando Kedro directamente
docker-compose exec airflow bash -c "cd /app && python -m kedro run --pipeline=feature_engineering"

# Verificar que los datos están montados
docker-compose exec airflow ls -lh /app/data/01_raw/
```

### Ver uso de recursos
```bash
# Ver estadísticas de CPU, memoria, etc.
docker stats clashroyale_ml_airflow
```

## 🔧 Solución de problemas

### Error: "port 8080 is already allocated"
```bash
# Ver qué está usando el puerto 8080
netstat -ano | findstr :8080

# Opción 1: Cambiar puerto en docker-compose.yml
# Cambia "8080:8080" a "8081:8080" y accede a http://localhost:8081

# Opción 2: Detener el proceso que usa el puerto
```

### Error: "build failed" o "no space left on device"
```bash
# Limpiar imágenes no usadas
docker system prune -a

# Ver espacio usado por Docker
docker system df
```

### Los DAGs no aparecen en Airflow
```bash
# Verificar que los DAGs están en el contenedor
docker-compose exec airflow ls -la /opt/airflow/dags/

# Revisar logs del scheduler
docker-compose logs airflow | grep -i dag

# Reiniciar el contenedor
docker-compose restart airflow
```

### Airflow no inicia
```bash
# Ver logs detallados
docker-compose logs airflow

# Verificar que la BD se inicializó
docker-compose exec airflow airflow db check

# Reconstruir desde cero
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## 🗄️ Comportamiento de la base de datos

**Importante**: La base de datos de Airflow se limpia automáticamente cada vez que inicias el contenedor. Esto significa:
- ✅ La BD se recrea desde cero en cada inicio
- ✅ El usuario admin siempre se crea con las credenciales configuradas (admin/admin)
- ✅ No se persisten datos históricos entre reinicios del contenedor
- ⚠️ Si necesitas persistir la BD, debes modificar `docker-entrypoint.sh` para no eliminar `airflow.db`

## 📊 Estructura de datos montados

Los datos **NO** están en la imagen Docker, sino que se montan como volumen:

```
./data (host)  →  /app/data (contenedor)
```

Esto significa:
- ✅ Cambios en datos del host se reflejan en el contenedor
- ✅ No necesitas rebuild si actualizas datos
- ✅ La imagen Docker es pequeña (~1-2 GB)
- ✅ Puedes versionar datos con DVC independientemente

## 🎯 Próximos pasos

Una vez que Airflow está corriendo:

1. **Ejecutar un DAG**: Haz clic en un DAG → "Trigger DAG"
2. **Monitorear ejecución**: Ve a la vista de árbol o gráfico
3. **Ver logs**: Clic en una tarea → "Log"
4. **Revisar resultados**: Los modelos y métricas se guardan en `data/06_models/` y `data/07_model_output/`

## 📝 Notas importantes

- **Primera vez**: El build puede tardar 10-15 minutos
- **Siguientes veces**: `docker-compose up -d` es casi instantáneo
- **Datos**: Asegúrate de tener los datos en `./data/` antes de ejecutar DAGs
- **Memoria**: Recomendado tener al menos 4GB RAM disponibles para Docker
- **Docker Desktop**: Mantén Docker Desktop corriendo siempre que uses el contenedor

