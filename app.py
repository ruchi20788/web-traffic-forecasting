import os, json, io, csv
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, send_file
from flask_cors import CORS
from forecast_utils import load_wide_csv, get_series_for_site, forecast_both, backtest_last_k

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(APP_DIR, "data", "web_traffic_sites_wideformat.csv")
CACHE_DIR = os.path.join(APP_DIR, "cache")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
CORS(app)

DEMO_ADMIN = {"username": "admin", "password": "admin123"}

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def get_WIDE():
    return load_wide_csv(DATA_CSV)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        if u == DEMO_ADMIN["username"] and p == DEMO_ADMIN["password"]:
            session["user"] = {"username": u}
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html")

@app.route("/api/sites")
@login_required
def api_sites():
    WIDE = get_WIDE()
    sites = list(WIDE.index)
    q = request.args.get("q", "").lower()
    if q:
        sites = [s for s in sites if q in s.lower()]
    return jsonify([{"id": s, "text": s} for s in sites])

def cache_path_for(site: str):
    safe = site.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")

@app.route("/api/train_all", methods=["POST"])
@login_required
def api_train_all():
    WIDE = get_WIDE()
    sites = list(WIDE.index)
    done = 0
    for s in sites:
        y = get_series_for_site(WIDE, s)
        fc = forecast_both(y, horizon=30)
        bt = backtest_last_k(y, k=14)
        out = {
            "site": s,
            "history": {"dates": [d.strftime("%Y-%m-%d") for d in y.index], "values": [int(v) for v in y.values]},
            "forecast": fc,
            "backtest": bt
        }
        with open(cache_path_for(s), "w", encoding="utf-8") as f:
            json.dump(out, f)
        done += 1
    return jsonify({"status": "ok", "trained": done})

@app.route("/api/forecast")
@login_required
def api_forecast():
    WIDE = get_WIDE()
    site = request.args.get("site")
    if not site or site not in WIDE.index:
        return jsonify({"error": "Valid 'site' required"}), 400
    cp = cache_path_for(site)
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    y = get_series_for_site(WIDE, site)
    hist_dates = [d.strftime("%Y-%m-%d") for d in y.index]
    hist_vals = [int(v) for v in y.values]
    fc = forecast_both(y, horizon=30)
    bt = backtest_last_k(y, k=14)
    return jsonify({
        "site": site,
        "history": {"dates": hist_dates, "values": hist_vals},
        "forecast": fc,
        "backtest": bt
    })

@app.route("/api/export_csv")
@login_required
def api_export_csv():
    WIDE = get_WIDE()
    site = request.args.get("site")
    if not site or site not in WIDE.index:
        return jsonify({"error": "Valid 'site' required"}), 400
    cp = cache_path_for(site)
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        y = get_series_for_site(WIDE, site)
        fc = forecast_both(y, horizon=30)
        payload = { "forecast": fc }
    fdates = payload["forecast"]["rf"]["dates"]
    rf = payload["forecast"]["rf"]["values"]
    sx = payload["forecast"]["sx"]["values"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "rf_forecast", "sarimax_forecast"])
    for d, r, s in zip(fdates, rf, sx):
        writer.writerow([d, r, s])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{site}_30day_forecast.csv"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
