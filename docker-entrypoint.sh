#!/bin/bash
set -e

# Función para inicializar Airflow (siempre limpia la BD)
init_airflow() {
    echo "=========================================="
    echo "Limpiando e inicializando base de datos de Airflow..."
    echo "=========================================="
    
    # Eliminar base de datos existente si existe
    if [ -f "${AIRFLOW_HOME}/airflow.db" ]; then
        echo "Eliminando base de datos existente: ${AIRFLOW_HOME}/airflow.db"
        rm -f ${AIRFLOW_HOME}/airflow.db
    fi
    
    # Eliminar también cualquier archivo de migraciones/reset previo
    echo "Limpiando archivos de configuración previos..."
    rm -f ${AIRFLOW_HOME}/simple_auth_manager_passwords.json.generated 2>/dev/null || true
    rm -f ${AIRFLOW_HOME}/webserver_config.py 2>/dev/null || true
    
    # En Airflow 3.x, usar 'migrate' para crear la BD desde cero
    echo "Creando nueva base de datos desde cero..."
    airflow db migrate
    echo "Base de datos inicializada correctamente."
    
    # Configurar usuario admin (siempre)
    # En Airflow 3.x, standalone usa simple_auth_manager_passwords.json.generated
    # Debemos crear este archivo ANTES de que standalone se inicie
    if [ -n "$AIRFLOW_ADMIN_USER" ] && [ -n "$AIRFLOW_ADMIN_PASSWORD" ]; then
        echo "Configurando usuario admin: ${AIRFLOW_ADMIN_USER}..."
        mkdir -p ${AIRFLOW_HOME}
        
        # Crear el archivo .generated que standalone usará
        # Formato: {"username": "password"}
        cat > ${AIRFLOW_HOME}/simple_auth_manager_passwords.json.generated << EOF
{"${AIRFLOW_ADMIN_USER}": "${AIRFLOW_ADMIN_PASSWORD}"}
EOF
        chmod 600 ${AIRFLOW_HOME}/simple_auth_manager_passwords.json.generated
        
        # También crear el archivo simple_auth_manager_passwords.json por si acaso
        cat > ${AIRFLOW_HOME}/simple_auth_manager_passwords.json << EOF
{
  "users": {
    "${AIRFLOW_ADMIN_USER}": "${AIRFLOW_ADMIN_PASSWORD}"
  }
}
EOF
        chmod 600 ${AIRFLOW_HOME}/simple_auth_manager_passwords.json
        echo "Usuario admin configurado correctamente: ${AIRFLOW_ADMIN_USER}/${AIRFLOW_ADMIN_PASSWORD}"
    else
        echo "Advertencia: AIRFLOW_ADMIN_USER o AIRFLOW_ADMIN_PASSWORD no están configurados."
    fi
}

# Inicializar Airflow
init_airflow

# Si el comando es 'standalone', ejecutarlo directamente
# Si no, ejecutar el comando pasado como argumento
if [ "$1" = "standalone" ]; then
    echo "Iniciando Airflow en modo standalone..."
    exec airflow standalone
else
    exec "$@"
fi

