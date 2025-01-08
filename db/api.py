from flask import Flask, request, jsonify

app = Flask(__name__)

# Emission factor for coal types (in tons CO2 per ton of coal)
emission_factors = {
    "bituminous": 2.6,
    "sub-bituminous": 2.4,
    "lignite": 1.9,
    "anthracite": 2.5
}

@app.route('/calculate_emissions', methods=['POST'])
def calculate_emissions():
    data = request.get_json()

    amount_mined = data.get('amount_mined')  # in tons
    coal_type = data.get('coal_type')  # e.g., 'bituminous'

    if amount_mined is None or coal_type not in emission_factors:
        return jsonify({"error": "Invalid input"}), 400

    emission_factor = emission_factors[coal_type]
    emissions = amount_mined * emission_factor  # in tons of CO2

    return jsonify({
        "amount_mined": amount_mined,
        "coal_type": coal_type,
        "emissions": emissions
    })

if __name__ == '__main__':
    app.run(debug=True)
