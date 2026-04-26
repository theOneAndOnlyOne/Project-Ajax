"""Leon S. Kennedy workout dashboard. Flask + Hevy.

  > "What's your business here, stranger?"
  > "Heard there's a workout to do."

Run:
    cp .env.example .env  # then fill in your HEVY_API_KEY
    pip install -r requirements.txt
    python app.py
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

import analytics
import supplements
from hevy_client import HevyError, from_env

load_dotenv(Path(__file__).parent / ".env")

app = Flask(__name__, static_folder="static", template_folder="templates")

LABEL = os.environ.get("LEON_LABEL", "v2")
PASSWORD = os.environ.get("LEON_PASSWORD", "").strip()
USERNAME = os.environ.get("LEON_USERNAME", "leon").strip()


@app.before_request
def _basic_auth():
    """Optional HTTP Basic Auth gate. Disabled when LEON_PASSWORD is unset."""
    if not PASSWORD:
        return None
    auth = request.authorization
    if (
        auth
        and hmac.compare_digest(auth.username or "", USERNAME)
        and hmac.compare_digest(auth.password or "", PASSWORD)
    ):
        return None
    return Response(
        "ACCESS DENIED // R.P.D. CLEARANCE REQUIRED",
        401,
        {"WWW-Authenticate": 'Basic realm="leon"'},
    )


def _client():
    return from_env()


@app.route("/")
def index():
    return render_template("index.html", label=LABEL)


@app.route("/api/workouts")
def api_workouts():
    label = request.args.get("label", LABEL)
    try:
        client = _client()
        all_w = client.all_workouts()
        filtered = client.filter_by_label(all_w, label)
        return jsonify(
            {
                "label": label,
                "total_in_account": len(all_w),
                "matched": len(filtered),
                "workouts": filtered,
            }
        )
    except HevyError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats")
def api_stats():
    label = request.args.get("label", LABEL)
    try:
        client = _client()
        all_w = client.all_workouts()
        filtered = client.filter_by_label(all_w, label)
        return jsonify(
            {
                "label": label,
                "stats": analytics.overall_stats(filtered),
                "volume_timeline": analytics.volume_timeline(filtered),
                "muscle_balance": analytics.muscle_balance(filtered),
            }
        )
    except HevyError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/overload")
def api_overload():
    label = request.args.get("label", LABEL)
    lookback = int(request.args.get("lookback", 5))
    try:
        client = _client()
        all_w = client.all_workouts()
        filtered = client.filter_by_label(all_w, label)
        history = analytics.per_exercise_history(filtered)
        return jsonify(
            {
                "label": label,
                "projections": analytics.progressive_overload(history, lookback=lookback),
            }
        )
    except HevyError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/supplements", methods=["GET"])
def api_supplements():
    return jsonify(supplements.get_state())


@app.route("/api/supplements/add", methods=["POST"])
def api_supplements_add():
    body = request.get_json(force=True) or {}
    field = body.get("field")
    amount = body.get("amount")
    if not field or amount is None:
        return jsonify({"error": "field and amount required"}), 400
    try:
        return jsonify(supplements.add_intake(field, float(amount)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/supplements/reset", methods=["POST"])
def api_supplements_reset():
    body = request.get_json(silent=True) or {}
    return jsonify(supplements.reset_today(body.get("field")))


@app.route("/api/supplements/targets", methods=["POST"])
def api_supplements_targets():
    body = request.get_json(force=True) or {}
    return jsonify(supplements.set_targets(body))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("LEON_PORT", 5000)))
    host = os.environ.get("LEON_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
