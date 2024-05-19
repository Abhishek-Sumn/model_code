from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

# Initialize the summarization pipeline
pipe = pipeline('summarization', model='t5-base')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the JSON data from the request body
    request_data = request.get_json()
    
    # Extract the text to be summarized
    input_text = request_data.get('text')
    
    if not input_text:
        return jsonify({"error": "Text field is required"}), 400

    # Generate the summary
    pipe_out = pipe(input_text)
    summary_text = pipe_out[0]['summary_text']

    # Return the summary as a JSON response
    return jsonify({"summary": summary_text})

if __name__ == '__main__':
    app.run(debug=True)
