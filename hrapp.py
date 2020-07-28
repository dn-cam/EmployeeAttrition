from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def add_message():
    content = request.json
    print(content)
    return content

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)