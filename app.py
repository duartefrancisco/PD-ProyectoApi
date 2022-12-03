from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
from termcolor import colored

app = Flask(__name__)

modelo = joblib.load("titanicSex_pipeline_v122022.pkl")
features = joblib.load("FEATURES.pkl")

def ActualizarLog(mensaje, info = True):
    archivoLog = open("logData.log", "a")
    mensajeCompleto = f"{'INFO' if info else 'ERROR'} - {mensaje} - {datetime.today().strftime('%Y-$%m-%d %H:%M:%S')};\n"
    archivoLog.write(mensajeCompleto)
    archivoLog.close()
    print(colored(mensajeCompleto, "green" if info else "red"))

@app.route("/predecir", methods=["POST"])
def predecir():
    ActualizarLog("Solicitud de predicción")
    data = request.get_json()
    dataframe = pd.json_normalize(data)
    ActualizarLog("Normalización JSON completada")

    ids = dataframe["PassengerId"]

    try:
        ActualizarLog("Se realiza las predicciones, Modelo: titanicSex_pipeline_v122022")
        predicciones = modelo.predict(dataframe)
        ActualizarLog("Finalizan las predicciones")
    except Exception as e:
        ActualizarLog(f"Ocurrió un error durante la predicción, {type(e)}: {e}", info= False)

    out = {}

    ActualizarLog("Prepara data para retornar")
    for index, item in enumerate(predicciones):
        out[ids[index]] = item

    ActualizarLog("Data lista para retornar")

    ActualizarLog("Se retorna resultado de la consulta")
    return jsonify(out)