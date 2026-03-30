import os

from io import BytesIO
from flask import send_file, jsonify, request

from functools import wraps
from io import StringIO

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    send_file,
)

import pandas as pd

from forecast_utils import (
    load_wide_csv,
    get_series_for_site,
    forecast_both,
    backtest_last_k,
)

DATA_PATH = os.path.join("data", "web_traffic_sites_wideformat.csv")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

DEMO_ADMIN = {
    "username": os.environ.get("ADMIN_USER", "admin"),
    "password": os.environ.get("ADMIN_PASS", "admin123"),
}

_WIDE = None


def get_wide():
    global _WIDE
    if _WIDE is None:
        _WIDE = load_wide_csv(DATA_PATH)
    return _WIDE


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == DEMO_ADMIN["username"] and password == DEMO_ADMIN["password"]:
            session["user"] = {"username": username}
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password")

    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    return render_template("index.html")


# ---------- API: site list for Select2 ----------
@app.route("/api/sites")
@login_required
def api_sites():
    wide = get_wide()
    q = (request.args.get("q") or "").lower()

    sites = [str(s) for s in wide.index]
    if q:
        sites = [s for s in sites if q in s.lower()]

    # Select2 expects [{id,text}, ...]
    return jsonify([{"id": s, "text": s} for s in sites])


# ---------- API: forecast ----------
@app.route("/api/forecast")
@login_required
def api_forecast():
    site = request.args.get("site")
    horizon = int(request.args.get("horizon", 30))
    k_backtest = int(request.args.get("k", 14))

    if not site:
        return jsonify({"error": "Missing 'site' parameter"}), 400

    wide = get_wide()
    if site not in wide.index:
        return jsonify({"error": f"Site '{site}' not found"}), 404

    y = get_series_for_site(wide, site)

    history_dates = [d.strftime("%Y-%m-%d") for d in y.index]
    history_values = [float(v) for v in y.values]

    fc = forecast_both(y, horizon=horizon)
    bt = backtest_last_k(y, k=k_backtest)

    return jsonify(
        {
            "site": site,
            "history": {
                "dates": history_dates,
                "values": history_values,
            },
            "forecast": fc,
            "backtest": bt,
        }
    )


# ---------- API: export CSV ----------

''' @app.route("/api/export_csv")
@login_required
def api_export_csv():
    site = request.args.get("site")
    horizon = int(request.args.get("horizon", 30))

    if not site:
        return jsonify({"error": "Missing 'site' parameter"}), 400

    wide = get_wide()
    if site not in wide.index:
        return jsonify({"error": f"Site '{site}' not found"}), 404

    y = get_series_for_site(wide, site)
    fc = forecast_both(y, horizon=horizon)

    df = pd.DataFrame(
        {
            "date_rf": fc["rf"]["dates"],
            "rf_forecast": fc["rf"]["values"],
        }
    )
    df["date_sx"] = fc["sx"]["dates"]
    df["sarimax_forecast"] = fc["sx"]["values"]
    df["date_lstm"] = fc["lstm"]["dates"]
    df["lstm_forecast"] = fc["lstm"]["values"]

    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    return send_file(
        csv_buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{site}_30d_forecast.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)'''


@app.route("/api/export_csv")
@login_required
def api_export_csv():
    site = request.args.get("site")
    horizon = int(request.args.get("horizon", 30))

    if not site:
        return jsonify({"error": "Missing 'site' parameter"}), 400

    wide = get_wide()
    if site not in wide.index:
        return jsonify({"error": f"Site '{site}' not found"}), 404

    y = get_series_for_site(wide, site)
    fc = forecast_both(y, horizon=horizon)

    df = pd.DataFrame({
        "date_rf": fc["rf"]["dates"],
        "rf_forecast": fc["rf"]["values"],
        "date_sarimax": fc["sx"]["dates"],
        "sarimax_forecast": fc["sx"]["values"],
        "date_lstm": fc["lstm"]["dates"],
        "lstm_forecast": fc["lstm"]["values"],
    })

    # ✅ Use BytesIO instead of StringIO
    csv_buf = BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    return send_file(
        csv_buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{site}_{horizon}d_forecast.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
