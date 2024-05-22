from flask import Flask, jsonify, request
from transformers import pipeline
from pyngrok import ngrok
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

port_no = 5000
ngrok.set_auth_token("2fpJeZSEIxJ9eqqEGoxMX2m4cxz_7t5i6NHUaU256iffW7USe")
public_url = ngrok.connect(port_no).public_url
print(f" * Ngrok Tunnel: {public_url}")

# Initialize the summarization pipeline
pipe = pipeline('summarization', model='t5-base', tokenizer='t5-base')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the JSON data from the request body
    request_data = request.get_json()
    
    # Extract the text to be summarized
    input_text = request_data.get('text')
    
    if not input_text:
        return jsonify({"error": "Text field is required"}), 400

    try:
        # Generate the summary
        pipe_out = pipe(input_text)
        summary_text = pipe_out[0]['summary_text']
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return the summary as a JSON response
    return jsonify({"summary": summary_text})

if __name__ == '__main__':
    app.run(port=port_no)
