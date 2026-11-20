"""Local web demo:  python3 app.py  ->  http://127.0.0.1:5057"""
import json, os, random, sys

from flask import Flask, jsonify, request, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify

app = Flask(__name__)


@app.get("/")
def index():
    return Response(open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8").read(),
                    mimetype="text/html")


@app.get("/api/info")
def info():
    """Accuracy shown in the UI = reported results (results.json), not the bundled single models' own test scores."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.post("/api/classify")
def api_classify():
    text = (request.get_json(force=True).get("text") or "").strip()
    if not text:
        return jsonify({"error": "Paste or type some text first."}), 400
    try:
        return jsonify(classify.analyze(text))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/sample")
def sample():
    s = random.choice(classify.bundle()["samples"])
    return jsonify(s)


if __name__ == "__main__":
    classify.bundle(); classify.basebundle(); classify.neuralbundle()
    import features; features._pipelines(); features.xpos_tags([["warm"]])   # warm up before the first request
    app.run(host="127.0.0.1", port=5057, debug=False)
