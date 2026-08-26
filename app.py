import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sqlite3
from datetime import datetime

# ============================================
# LOAD MODEL
# ============================================
class_names = ["good", "bent", "color", "flip", "scratch"]  # update to match YOUR classes

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('best_defect_model.pth', map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# SQL LOGGING
# ============================================
def init_db():
    conn = sqlite3.connect('defect_logs.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS inspections
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     prediction TEXT,
                     confidence REAL,
                     is_defective INTEGER,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

def log_inspection(prediction, confidence, is_defective):
    conn = sqlite3.connect('defect_logs.db')
    conn.execute("INSERT INTO inspections (prediction, confidence, is_defective) VALUES (?, ?, ?)",
                 (prediction, confidence, is_defective))
    conn.commit()
    conn.close()

# ============================================
# PREDICT
# ============================================
def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()

# ============================================
# STREAMLIT UI
# ============================================
def main():
    st.set_page_config(page_title="ManuVision AI", page_icon="🔍", layout="wide")
    st.title("🔍 ManuVision AI")
    st.subheader("AI-Powered Manufacturing Defect Detection")

    model = load_model()
    init_db()

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Upload Product Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if uploaded_file:
            st.write("### Inspection Result")
            prediction, confidence = predict(image, model)
            is_defective = 0 if prediction == "good" else 1

            if is_defective:
                st.error(f"⚠️ DEFECT DETECTED: **{prediction.upper()}**")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                st.write(f"**Defect Type:** {prediction}")
                st.write("**Recommended Action:** Quarantine part for further inspection")
            else:
                st.success(f"✅ PART OK — No defect detected")
                st.metric("Confidence", f"{confidence*100:.1f}%")

            # Log to database
            log_inspection(prediction, confidence, is_defective)

    # ============================================
    # DASHBOARD — defect trends
    # ============================================
    st.write("---")
    st.write("### 📊 Inspection Dashboard")

    conn = sqlite3.connect('defect_logs.db')
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    defective = conn.execute("SELECT COUNT(*) FROM inspections WHERE is_defective = 1").fetchone()[0]
    good = total - defective

    if total > 0:
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("Total Inspections", total)
        dcol2.metric("Defective", defective, delta=f"{defective/total*100:.0f}%", delta_color="inverse")
        dcol3.metric("Pass Rate", f"{good/total*100:.1f}%")

        # Defect type breakdown
        rows = conn.execute("SELECT prediction, COUNT(*) FROM inspections WHERE is_defective = 1 GROUP BY prediction").fetchall()
        if rows:
            st.write("**Defect Type Breakdown:**")
            for row in rows:
                st.write(f"- {row[0]}: {row[1]} occurrences")

    # Recent inspections
    st.write("### Recent Inspections")
    recent = conn.execute("SELECT prediction, confidence, is_defective, timestamp FROM inspections ORDER BY timestamp DESC LIMIT 10").fetchall()
    for row in recent:
        status = "❌ DEFECT" if row[2] else "✅ GOOD"
        st.write(f"{status} | {row[0]} | {row[1]*100:.1f}% confidence | {row[3]}")

    conn.close()

if __name__ == "__main__":
    main()