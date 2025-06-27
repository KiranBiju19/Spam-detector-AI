from flask import Flask, render_template_string, request
from predictor import predict_message

app = Flask(__name__)

HTML = '''
<!doctype html>
<title>Spam/Phishing Detector</title>
<h2>Spam/Phishing Message Detector</h2>
<form method=post>
  <textarea name=message rows=6 cols=60 placeholder="Enter your message here..."></textarea><br>
  <input type=submit value=Predict>
</form>
{% if prediction %}
  <h3>Prediction: {{ prediction }}</h3>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        prediction = predict_message(message)
    return render_template_string(HTML, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 