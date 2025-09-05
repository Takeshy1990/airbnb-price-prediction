# make_report.py — Portfolio-ready PDF με metrics, predictions, feature importances & plot
from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

BASE = Path(".").resolve()
META_FILE = BASE / "airbnb_meta.json"
PRED_FILE = BASE / "prediction.csv"
FEAT_FILE = BASE / "feature_importance.csv"   # αν το αρχείο σου λέγεται αλλιώς, άλλαξέ το εδώ
PLOT_FILE = BASE / "prediction_plot.png"
PDF_FILE  = BASE / "airbnb_report.pdf"

def load_csv_any(path: Path) -> pd.DataFrame:
    for sep in (None, ",", ";", "\t"):
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            pass
    raise FileNotFoundError(f"Δεν μπόρεσα να διαβάσω: {path}")

def df_to_table(df: pd.DataFrame, max_rows=20):
    df_ = df.head(max_rows).copy()
    data = [list(df_.columns)] + df_.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return table

def main():
    styles = getSampleStyleSheet()
    story = []

    # Τίτλος
    story.append(Paragraph("<b>Airbnb Price Prediction — Portfolio Report</b>", styles["Title"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("End-to-end ML pipeline: data cleaning, feature engineering, "
                           "Gradient Boosting (median + quantiles), evaluation & batch predictions.",
                           styles["Normal"]))
    story.append(Spacer(1, 18))

    # Metrics
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    cv = meta["cv"]
    story.append(Paragraph("<b>Model Performance (5-fold Cross-Validation)</b>", styles["Heading2"]))
    story.append(Paragraph(f"R²: {cv['R2']:.3f} — MAE: {cv['MAE']:.1f} € — RMSE: {cv['RMSE']:.1f} €",
                           styles["Normal"]))
    story.append(Spacer(1, 14))

    # Predictions
    if PRED_FILE.exists():
        pred = load_csv_any(PRED_FILE)
        story.append(Paragraph("<b>Batch Predictions (median & q10–q90)</b>", styles["Heading2"]))
        story.append(df_to_table(pred))
        story.append(Spacer(1, 14))

    # Feature importances
    if FEAT_FILE.exists():
        feat = load_csv_any(FEAT_FILE).sort_values("importance", ascending=False)
        # Αν το αρχείο σου έχει άλλο όνομα (π.χ. "features.csv"), μετονόμασέ το σε feature_importance.csv ή άλλαξε το FEAT_FILE πιο πάνω
        story.append(Paragraph("<b>Top Feature Importances</b>", styles["Heading2"]))
        feat["importance"] = feat["importance"].round(6)
        story.append(df_to_table(feat[["feature","importance"]], max_rows=20))
        story.append(Spacer(1, 14))

    # Plot
    if PLOT_FILE.exists():
        story.append(Paragraph("<b>Actual vs Predicted</b>", styles["Heading2"]))
        story.append(Image(str(PLOT_FILE), width=400, height=400))
        story.append(Spacer(1, 14))

    # Short interpretation
    story.append(Paragraph(
        "Σχόλιο: Το μοντέλο δείχνει υψηλή επεξηγητική ισχύ. "
        "Η μεταβλητή <i>review_scores_rating</i> είναι ο ισχυρότερος driver, "
        "ενώ ο τύπος δωματίου και η διαθεσιμότητα επηρεάζουν έντονα το pricing. "
        "Τα quantile μοντέλα (q10–q90) παρέχουν ρεαλιστικά εύρη τιμολόγησης.",
        styles["Italic"]
    ))

    # Save PDF
    doc = SimpleDocTemplate(str(PDF_FILE))
    doc.build(story)
    print(f"Report saved -> {PDF_FILE}")

if __name__ == "__main__":
    main()