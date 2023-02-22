from flask import Flask, request, send_file


import asthmaStage
import barChart

app = Flask(__name__)


@app.route("/stage", methods=["GET"])
def stage():

    json_data = request.get_json()
    print('Incoming JSON data:', json_data)
    stageData = json_data['stage']
    print(stageData)
    num1 = stageData[0]
    num2 = stageData[1]
    num3 = stageData[2]
    num4 = stageData[3]
    print(num1," ",num2," ",num3," ",num4)
    result = asthmaStage.predictData(num1, num2, num3, num4)
    print("Prediction is", result)
    return {"stage": int(result)}


@app.route('/plot', methods=['GET'])
def plot():
    json_data = request.get_json()
    print('Incoming JSON data:', json_data)
    # process the JSON data here
    plot_data = json_data['plot'][:4]
    result = barChart.genarateBarChart(plot_data)
    print(result, "Results in test json")

    return {"plotData": result}


if __name__ == "__main__":
    app.run(debug=True)
