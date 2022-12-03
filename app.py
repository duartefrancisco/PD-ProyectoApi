from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
from termcolor import colored

app = Flask(__name__)

versionModelo = "titanicSex_pipeline_v122022.pkl"
modelo = joblib.load(versionModelo)
features = joblib.load("FEATURES.pkl")

def ActualizarLog(mensaje, info = True):
    archivoLog = open("logData.log", "a")
    mensajeCompleto = f"{'INFO' if info else 'ERROR'} - {mensaje} - {datetime.today().strftime('%Y-%m-%d %H:%M:%S')};\n"
    archivoLog.write(mensajeCompleto)
    archivoLog.close()
    print(colored(mensajeCompleto, "green" if info else "red"))

@app.route("/predecir", methods=["POST"])
def predecir():
    ActualizarLog("Solicitud de predicción")
    ActualizarLog("Inicia Request JSON")
    data = request.get_json()
    ActualizarLog("Finaliza Request JSON")
    ActualizarLog("Inicia Normalización JSON")
    dataframe = pd.json_normalize(data)
    ActualizarLog("Termina Normalización JSON")

    ids = dataframe["PassengerId"]

    try:
        ActualizarLog(f"Inicia predicciones, Modelo: {versionModelo}")
        predicciones = modelo.predict(dataframe)
        ActualizarLog("Finalizan las predicciones")
        
        out = {}

        ActualizarLog("Prepara data para retornar")
        for index, item in enumerate(predicciones):
            out[str(ids[index])] = int(item)

        ActualizarLog("Data lista para retornar")

        ActualizarLog("Se retorna resultado de las predicciones")        
        return jsonify(out)
    except Exception as e:
        error = f"Ocurrió un error, {type(e)}: {e}"
        ActualizarLog(error, info= False)
        ActualizarLog("Se retorna mensaje de error")        
        return jsonify({"mensaje": error})
