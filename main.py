from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import json
import os
import io
import csv
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="API de Gestión de Empresas", version="1.0.0")

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # permite frontend
    allow_credentials=True,
    allow_methods=["*"],     # permite todos los métodos, incluido OPTIONS
    allow_headers=["*"],
)

DB_FILE = 'industria_map.json'
VENTAS_FILE = "ventas.csv"

# Modelos Pydantic para validación de datos
class Empresa(BaseModel):
    empresa: str
    industria: str

class EmpresaUpdate(BaseModel):
    industria: str

# Función para cargar datos del archivo JSON
def cargar_datos() -> Dict[str, str]:
    if not os.path.exists(DB_FILE):
        # Si el archivo no existe, crear uno con datos iniciales
        datos_iniciales = {
            "cmpc": "forestal_construccion",
            "lippi": "textil",
            "petco": "mascotas"
        }
        guardar_datos(datos_iniciales)
        return datos_iniciales
    
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def guardar_datos(datos: Dict[str, str]):
    with open(DB_FILE, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=2, ensure_ascii=False)

# Función para validar formato de fecha
def validar_fecha(fecha_str: str) -> bool:
    try:
        datetime.strptime(fecha_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# Función para validar que el cliente existe en empresas.json
def validar_cliente(cliente: str, empresas: Dict[str, str]) -> bool:
    return cliente.lower() in [empresa.lower() for empresa in empresas.keys()]

# Función para detectar el separador CSV
def detectar_separador(primera_linea: str) -> str:
    # Contar ocurrencias de posibles separadores
    comas = primera_linea.count(',')
    puntos_coma = primera_linea.count(';')
    
    # Si hay más punto y coma que comas, usar punto y coma
    if puntos_coma > comas:
        return ';'
    else:
        return ','

# Función para validar y procesar archivo CSV
def procesar_csv(file_content: str) -> List[Dict]:
    lines = file_content.strip().split('\n')
    if not lines:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    
    # Detectar el separador automáticamente
    separador = detectar_separador(lines[0])
    
    # Verificar encabezados
    headers = [h.strip().lower() for h in lines[0].split(separador)]
    expected_headers = ['fecha', 'cliente', 'unidades']
    
    if headers != expected_headers:
        raise HTTPException(
            status_code=400, 
            detail=f"Encabezados incorrectos. Se esperan: {expected_headers}, se encontraron: {headers}. Separador detectado: '{separador}'"
        )
    
    empresas = cargar_datos()
    registros_validos = []
    errores = []
    
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():  # Saltar líneas vacías
            continue
            
        try:
            parts = [p.strip() for p in line.split(separador)]
            if len(parts) != 3:
                errores.append(f"Línea {i}: Número incorrecto de columnas (esperadas: 3, encontradas: {len(parts)})")
                continue
            
            fecha, cliente, unidades_str = parts
            
            # Validar fecha
            if not validar_fecha(fecha):
                errores.append(f"Línea {i}: Formato de fecha inválido '{fecha}'. Use YYYY-MM-DD")
                continue
            
            # Validar cliente
            if not validar_cliente(cliente, empresas):
                errores.append(f"Línea {i}: Cliente '{cliente}' no existe en la base de datos de empresas")
                continue
            
            # Validar unidades (manejar posibles decimales con coma)
            try:
                # Reemplazar coma decimal por punto si existe
                unidades_str_normalizado = unidades_str.replace(',', '.')
                unidades_float = float(unidades_str_normalizado)
                unidades = int(unidades_float)  # Convertir a entero
                
                # Verificar que no se perdió información decimal significativa
                if abs(unidades_float - unidades) > 0.001:
                    errores.append(f"Línea {i}: Unidades '{unidades_str}' debe ser un número entero")
                    continue
                    
            except ValueError:
                errores.append(f"Línea {i}: Unidades '{unidades_str}' debe ser un número entero")
                continue
            
            registros_validos.append({
                'fecha': fecha,
                'cliente': cliente,
                'unidades': unidades
            })
            
        except Exception as e:
            errores.append(f"Línea {i}: Error procesando línea - {str(e)}")
    
    if errores:
        raise HTTPException(
            status_code=400, 
            detail={
                "mensaje": "Se encontraron errores en el archivo CSV",
                "errores": errores,
                "registros_procesados": len(registros_validos),
                "separador_detectado": separador
            }
        )
    
    return registros_validos

# Función para guardar datos de ventas
def guardar_ventas(registros: List[Dict]):
    df = pd.DataFrame(registros)
    df.to_csv(VENTAS_FILE, index=False, encoding='utf-8')

# Función para cargar datos de ventas
def cargar_ventas() -> List[Dict]:
    if not os.path.exists(VENTAS_FILE):
        return []
    
    try:
        df = pd.read_csv(VENTAS_FILE, encoding='utf-8')
        return df.to_dict('records')
    except Exception:
        return []


@app.get("/empresas", summary="Obtener todas las empresas")
async def obtener_empresas():
    """
    Obtiene la lista completa de empresas con sus respectivas industrias.
    """
    datos = cargar_datos()
    return {
        "total": len(datos),
        "empresas": datos
    }

# Endpoint 2: Agregar una nueva empresa
@app.post("/empresas", summary="Agregar nueva empresa")
async def agregar_empresa(empresa: Empresa):
    """
    Agrega una nueva empresa con su industria.
    """
    datos = cargar_datos()
    
    # Verificar si la empresa ya existe
    if empresa.empresa.lower() in [key.lower() for key in datos.keys()]:
        raise HTTPException(
            status_code=400, 
            detail=f"La empresa '{empresa.empresa}' ya existe"
        )
    
    # Agregar la nueva empresa
    datos[empresa.empresa.lower()] = empresa.industria
    guardar_datos(datos)
    
    return {
        "mensaje": f"Empresa '{empresa.empresa}' agregada exitosamente",
        "empresa": empresa.empresa,
        "industria": empresa.industria
    }

# Endpoint 3: Editar una empresa existente
@app.put("/empresas/{nombre_empresa}", summary="Editar empresa existente")
async def editar_empresa(nombre_empresa: str, empresa_update: EmpresaUpdate):
    """
    Edita la industria de una empresa existente.
    """
    datos = cargar_datos()
    
    # Verificar si la empresa existe
    empresa_key = None
    for key in datos.keys():
        if key.lower() == nombre_empresa.lower():
            empresa_key = key
            break
    
    if not empresa_key:
        raise HTTPException(
            status_code=404, 
            detail=f"La empresa '{nombre_empresa}' no existe"
        )
    
    # Actualizar la industria
    datos[empresa_key] = empresa_update.industria
    guardar_datos(datos)
    
    return {
        "mensaje": f"Empresa '{empresa_key}' actualizada exitosamente",
        "empresa": empresa_key,
        "nueva_industria": empresa_update.industria
    }

# Endpoint 4: Eliminar una empresa
@app.delete("/empresas/{nombre_empresa}", summary="Eliminar empresa")
async def eliminar_empresa(nombre_empresa: str):
    """
    Elimina una empresa de la base de datos.
    """
    datos = cargar_datos()
    
    # Verificar si la empresa existe
    empresa_key = None
    for key in datos.keys():
        if key.lower() == nombre_empresa.lower():
            empresa_key = key
            break
    
    if not empresa_key:
        raise HTTPException(
            status_code=404, 
            detail=f"La empresa '{nombre_empresa}' no existe"
        )
    
    # Eliminar la empresa
    industria_eliminada = datos.pop(empresa_key)
    guardar_datos(datos)
    
    return {
        "mensaje": f"Empresa '{empresa_key}' eliminada exitosamente",
        "empresa_eliminada": empresa_key,
        "industria": industria_eliminada
    }

# Endpoint 5: Obtener una empresa específica
@app.get("/empresas/{nombre_empresa}", summary="Obtener empresa específica")
async def obtener_empresa(nombre_empresa: str):
    """
    Obtiene información de una empresa específica.
    """
    datos = cargar_datos()
    
    # Buscar la empresa (case insensitive)
    for key, value in datos.items():
        if key.lower() == nombre_empresa.lower():
            return {
                "empresa": key,
                "industria": value
            }
    
    raise HTTPException(
        status_code=404, 
        detail=f"La empresa '{nombre_empresa}' no existe"
    )

# Endpoint de salud de la API
@app.get("/", summary="Estado de la API")
async def root():
    """
    Endpoint de salud para verificar que la API está funcionando.
    """
    return {"mensaje": "API de Gestión de Empresas funcionando correctamente"}


# ENDPOINTS PARA MANEJO DE VENTAS CSV

# Endpoint para cargar datos CSV
@app.post("/ventas/cargar", summary="Cargar datos de ventas desde CSV")
async def cargar_ventas_csv(file: UploadFile = File(...)):
    """
    Carga datos de ventas desde un archivo CSV.
    El archivo debe tener las columnas: fecha, cliente, unidades
    
    Acepta tanto separadores de coma (,) como punto y coma (;)
    - fecha: formato YYYY-MM-DD
    - cliente: debe existir en la base de datos de empresas
    - unidades: número entero
    
    La API detecta automáticamente el separador utilizado.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        registros_validos = procesar_csv(content_str)
        
        if not registros_validos:
            raise HTTPException(status_code=400, detail="No se encontraron registros válidos en el archivo")
        
        # Guardar los datos (esto reemplaza los datos anteriores)
        guardar_ventas(registros_validos)
        
        return {
            "mensaje": "Datos de ventas cargados exitosamente",
            "registros_cargados": len(registros_validos),
            "archivo": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

# Endpoint para ver datos de ventas actuales
@app.get("/ventas", summary="Obtener datos de ventas actuales")
async def obtener_ventas():
    """
    Obtiene todos los datos de ventas cargados actualmente.
    """
    ventas = cargar_ventas()
    
    if not ventas:
        return {
            "mensaje": "No hay datos de ventas cargados",
            "total_registros": 0,
            "ventas": []
        }
    
    # Estadísticas adicionales
    total_unidades = sum(venta['unidades'] for venta in ventas)
    clientes_unicos = len(set(venta['cliente'] for venta in ventas))
    
    return {
        "total_registros": len(ventas),
        "total_unidades": total_unidades,
        "clientes_unicos": clientes_unicos,
        "ventas": ventas
    }

# Endpoint para descargar CSV de ejemplo
@app.get("/ventas/ejemplo", summary="Descargar CSV de ejemplo")
async def descargar_ejemplo_csv(separador: str = ","):
    """
    Descarga un archivo CSV de ejemplo con el formato correcto.
    Parámetros:
    - separador: ',' para formato internacional, ';' para formato español (por defecto: ',')
    """
    if separador not in [",", ";"]:
        raise HTTPException(status_code=400, detail="Separador debe ser ',' o ';'")
    
    empresas = cargar_datos()
    
    if not empresas:
        raise HTTPException(
            status_code=400, 
            detail="No hay empresas registradas. Agregue empresas primero."
        )
    
    # Crear datos de ejemplo
    ejemplos = [
        ["fecha", "cliente", "unidades"],
        ["2024-01-15", list(empresas.keys())[0], "100"],
        ["2024-01-16", list(empresas.keys())[0], "150"],
    ]
    
    # Si hay más empresas, agregar ejemplos con ellas
    if len(empresas) > 1:
        ejemplos.append(["2024-01-17", list(empresas.keys())[1], "200"])
    
    if len(empresas) > 2:
        ejemplos.append(["2024-01-18", list(empresas.keys())[2], "75"])
    
    # Crear CSV en memoria con el separador especificado
    output = io.StringIO()
    writer = csv.writer(output, delimiter=separador)
    writer.writerows(ejemplos)
    csv_content = output.getvalue()
    output.close()
    
    # Nombre del archivo según el separador
    filename = f"ejemplo_ventas_{'español' if separador == ';' else 'internacional'}.csv"
    
    # Crear respuesta de descarga
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)