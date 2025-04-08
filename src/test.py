from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/energy-stats')
def energy_stats():
    # Simulated energy stats for 10 epochs
    print("✅ /energy-stats route was hit")
    data = [
        {"epoch": 0, "energy": 0.1},
        {"epoch": 1, "energy": 0.3},
        {"epoch": 2, "energy": 0.5},
        {"epoch": 3, "energy": 0.7},
        {"epoch": 4, "energy": 1.0},
        {"epoch": 5, "energy": 1.3},
        {"epoch": 6, "energy": 1.6},
        {"epoch": 7, "energy": 1.9},
        {"epoch": 8, "energy": 2.1},
        {"epoch": 9, "energy": 2.4}
    ]
    print(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
