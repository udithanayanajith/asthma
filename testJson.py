from flask import Flask, request
import barChart

app = Flask(__name__)


@app.route('/myroute', methods=['GET'])
def handle_get_request():
    json_data = request.get_json()
    print('Incoming JSON data:', json_data)
    # process the JSON data here
    plot_data = json_data['plot'][:4]
    result = barChart.genarateBarChart(plot_data)
    print(result, "Results in test json")

    return {"plotData": result}


if __name__ == '__main__':
    app.run()
