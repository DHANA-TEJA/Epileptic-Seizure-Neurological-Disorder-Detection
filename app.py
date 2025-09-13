import os
import io
import csv
import uuid
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from src.dl_models import CNN_LSTM
from src.preprocess_input import load_eeg_file_for_gui

# ---------- CONFIG ----------
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results/gui"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
BATCH_TEST_FOLDER = r"C:\Users\Dhana Teja\OneDrive\Desktop\Individual_patients_excel_files"  # change if needed

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-for-production"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- LOAD MODEL ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM().to(device)
model_path = "results/saved_models/cnn_lstm_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Train/save or put model there.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- HELPERS ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_segments(segments, device=device):
    """segments: numpy array shape (N,1,256) -> returns probs and preds arrays"""
    import torch
    X = torch.tensor(segments, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(X)                          # logits (N,2)
        probs = torch.softmax(out, dim=1)[:,1]  # seizure probability (N,)
        preds = (probs >= 0.5).long()          # default binary threshold per-segment
    return probs.cpu().numpy(), preds.cpu().numpy()

def save_plot_and_highlight(signal, seg_indices, sample_rate=256, filename=None):
    """Plot signal (1D) and highlight segments indices (each segment length 256)."""
    seg_len = 256
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(12,3))
    plt.plot(t, signal, lw=0.8)
    for si in seg_indices:
        start = si * seg_len
        end = start + seg_len
        if start >= len(signal): 
            continue
        plt.axvspan(start / sample_rate, min(end, len(signal)) / sample_rate, color="red", alpha=0.25)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    fname = filename or f"{uuid.uuid4().hex}_plot.png"
    fpath = os.path.join(RESULTS_FOLDER, fname)
    plt.savefig(fpath)
    plt.close()
    return fpath

def segments_from_signal(signal, seg_len=256):
    segs = []
    for i in range(0, len(signal) - seg_len + 1, seg_len):
        segs.append(signal[i:i+seg_len])
    return np.array(segs)[:, np.newaxis, :]

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # file upload handling
        if "file" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(saved_path)
            # preprocess - returns numeric_df (pandas), channel_names (list)
            try:
                numeric_df, channel_names = load_eeg_file_for_gui(saved_path)
            except Exception as e:
                flash(f"Error reading file: {e}", "danger")
                return redirect(request.url)

            # store preprocessed in session-like file
            uid = uuid.uuid4().hex
            temp_csv = os.path.join(UPLOAD_FOLDER, f"{uid}_preview.csv")
            numeric_df.head(200).to_csv(temp_csv, index=False)  # small preview

            return render_template(
                "index.html",
                uploaded=True,
                filename=filename,
                preview_csv=os.path.basename(temp_csv),
                channels=channel_names
            )
        else:
            flash("Unsupported file format. Upload CSV or XLSX.", "warning")
            return redirect(request.url)
    return render_template("index.html", uploaded=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives form with:
     - filename (uploaded file saved earlier)
     - channel (name) or 'average'
     - probability_threshold (0-1) (optional)
     - percent_segments_threshold (0-1) : if fraction of segments exceed per-segment prob threshold -> flag
    """
    filename = request.form.get("filename")
    sel_channel = request.form.get("channel")
    avg_channels = request.form.get("avg_channels") == "on"
    per_seg_prob_threshold = float(request.form.get("prob_th", 0.5))
    seg_fraction_threshold = float(request.form.get("seg_frac", 0.3))

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        flash("Uploaded file not found on server. Please re-upload.", "danger")
        return redirect(url_for("index"))

    # load numeric df again (safe)
    numeric_df, channel_names = load_eeg_file_for_gui(input_path)
    if numeric_df.shape[1] == 0:
        flash("No numeric channels found in file.", "danger")
        return redirect(url_for("index"))

    # pick signal
    if avg_channels:
        signal = numeric_df.mean(axis=1).values
        chosen_label = "Average of channels"
    else:
        if sel_channel not in numeric_df.columns:
            flash("Selected channel not found. Using first channel.", "warning")
            sel_channel = numeric_df.columns[0]
        signal = numeric_df[sel_channel].values
        chosen_label = sel_channel

    # make segments
    seg_len = 256
    segments = segments_from_signal(signal, seg_len=seg_len)
    if segments.size == 0:
        flash(f"Signal too short for 1 segment of {seg_len} samples.", "danger")
        return redirect(url_for("index"))

    # pad last segment if needed (if you want to include tail)
    # (here we only used full-length segments above)

    # predict per-segment probabilities
    probs, preds = predict_segments(segments)

    # per-segment boolean using per_seg_prob_threshold
    seg_pos = (probs >= per_seg_prob_threshold).astype(int)
    seizure_fraction = seg_pos.sum() / max(1, len(seg_pos))
    seizure_detected = seizure_fraction >= seg_fraction_threshold

    # save per-segment table to CSV
    table_rows = []
    for i, (p, pr) in enumerate(zip(probs, seg_pos)):
        table_rows.append({"segment_index": i, "seizure_prob": float(p), "is_seizure": int(pr)})

    table_fname = f"{uuid.uuid4().hex}_preds.csv"
    table_path = os.path.join(RESULTS_FOLDER, table_fname)
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_index", "seizure_prob", "is_seizure"])
        writer.writeheader()
        writer.writerows(table_rows)

    # plot with highlights for segments detected
    highlighted_indices = [r["segment_index"] for r in table_rows if r["is_seizure"] == 1]
    plot_path = save_plot_and_highlight(signal, highlighted_indices, filename=f"{uuid.uuid4().hex}_plot.png")

    # result text and confidence
    result_text = "Seizure Detected ✅" if seizure_detected else "No Seizure ❌"
    confidence = float(seizure_fraction)  # fraction of segments flagged

    return render_template(
        "result.html",
        filename=filename,
        chosen_channel=chosen_label,
        result_text=result_text,
        confidence=f"{confidence:.2f}",
        plot_url=os.path.relpath(plot_path, start="."),
        table_url=os.path.relpath(table_path, start="."),
        table_rows=table_rows,
        thresholds={"per_seg": per_seg_prob_threshold, "seg_frac": seg_fraction_threshold}
    )


@app.route("/batch", methods=["GET"])
def batch():
    # scans BATCH_TEST_FOLDER for files and runs evaluate_file quickly (no upload)
    results = []
    for fname in os.listdir(BATCH_TEST_FOLDER):
        if not allowed_file(fname):
            continue
        fpath = os.path.join(BATCH_TEST_FOLDER, fname)
        try:
            numeric_df, _ = load_eeg_file_for_gui(fpath)
            # choose first channel average
            sig = numeric_df.mean(axis=1).values
            segs = segments_from_signal(sig)
            if segs.size == 0:
                results.append({"filename": fname, "error": "too short"})
                continue
            probs, preds = predict_segments(segs)
            seizure_fraction = (probs >= 0.5).mean()
            detected = seizure_fraction >= 0.3
            results.append({
                "filename": fname,
                "seizure_fraction": f"{seizure_fraction:.3f}",
                "detected": "Yes" if detected else "No",
                "segments": len(segs)
            })
        except Exception as e:
            results.append({"filename": fname, "error": str(e)})
    return render_template("batch_result.html", results=results)

@app.route("/download/<path:filename>")
def download_file(filename):
    path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(path):
        flash("File not found.", "danger")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
