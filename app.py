import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sqlite3
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# ============================================
# CONFIG
# ============================================
class_names = ["good", "bent", "color", "flip", "scratch"]  # update to match YOUR classes

# ============================================
# U-NET MODEL (for segmentation)
# ============================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_classifier():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('best_defect_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_segmentation_model():
    model = UNet()
    model.load_state_dict(torch.load('best_segmentation_model.pth', map_location='cpu'))
    model.eval()
    return model

# ============================================
# TRANSFORMS
# ============================================
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segment_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
                     defect_area_mm2 REAL,
                     defect_length_mm REAL,
                     within_tolerance INTEGER,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

def log_inspection(prediction, confidence, is_defective, area=0, length=0, within_tol=1):
    conn = sqlite3.connect('defect_logs.db')
    conn.execute(
        "INSERT INTO inspections (prediction, confidence, is_defective, defect_area_mm2, defect_length_mm, within_tolerance) VALUES (?, ?, ?, ?, ?, ?)",
        (prediction, confidence, is_defective, area, length, within_tol)
    )
    conn.commit()
    conn.close()

# ============================================
# PREDICTION FUNCTIONS
# ============================================
def classify(image, model):
    img_tensor = classify_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()

def segment_defect(image, model):
    img_tensor = segment_transform(image).unsqueeze(0)
    with torch.no_grad():
        mask = model(img_tensor)
    mask = mask.squeeze().numpy()

    # Threshold
    mask = (mask > 0.25).astype(np.uint8)
    
    # POST-PROCESSING (same idea as your BraTS project!)
    from scipy.ndimage import binary_opening, binary_closing
    
    # Remove small noise (opening = erode then dilate)
    mask = binary_opening(mask, structure=np.ones((5,5))).astype(np.uint8)
    
    # Fill small holes (closing = dilate then erode)
    mask = binary_closing(mask, structure=np.ones((3,3))).astype(np.uint8)
    
    # Remove regions smaller than 50 pixels (noise)
    labeled, num_features = ndimage.label(mask)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < 50:
            mask[labeled == i] = 0
    
    return mask

def measure_defect(mask, pixel_size_mm=0.1):
    labeled, num_features = ndimage.label(mask)
    measurements = []
    for i in range(1, num_features + 1):
        region = (labeled == i)
        area_pixels = region.sum()
        area_mm2 = area_pixels * (pixel_size_mm ** 2)
        coords = np.where(region)
        if len(coords[0]) == 0:
            continue
        height = (coords[0].max() - coords[0].min()) * pixel_size_mm
        width = (coords[1].max() - coords[1].min()) * pixel_size_mm
        max_length = max(height, width)
        measurements.append({
            'area_mm2': round(area_mm2, 2),
            'max_length_mm': round(max_length, 2),
            'height_mm': round(height, 2),
            'width_mm': round(width, 2),
            'area_pixels': int(area_pixels)
        })
    return measurements

def check_tolerance(measurements, tolerances):
    results = []
    for m in measurements:
        passed = True
        reasons = []
        if m['max_length_mm'] > tolerances.get('max_length_mm', float('inf')):
            passed = False
            reasons.append(f"Length {m['max_length_mm']}mm exceeds limit {tolerances['max_length_mm']}mm")
        if m['area_mm2'] > tolerances.get('max_area_mm2', float('inf')):
            passed = False
            reasons.append(f"Area {m['area_mm2']}mm² exceeds limit {tolerances['max_area_mm2']}mm²")
        results.append({
            'measurement': m,
            'passed': passed,
            'reasons': reasons if reasons else ['Within tolerance']
        })
    return results

# ============================================
# STREAMLIT APP
# ============================================
def main():
    st.set_page_config(page_title="ManuVision AI", page_icon="🔍", layout="wide")
    st.title("🔍 ManuVision AI")
    st.subheader("AI-Powered Manufacturing Defect Detection & Analysis")

    # Load models
    classifier = load_classifier()
    
    # Check if segmentation model exists
    seg_model_exists = True
    try:
        seg_model = load_segmentation_model()
    except:
        seg_model_exists = False
        seg_model = None

    init_db()

    # ---- SIDEBAR: Tolerance Settings ----
    st.sidebar.header("⚙️ Batch Tolerance Settings")
    st.sidebar.write("Configure per-batch quality limits:")
    max_length = st.sidebar.number_input("Max defect length (mm)", value=2.0, min_value=0.1, step=0.1)
    max_area = st.sidebar.number_input("Max defect area (mm²)", value=5.0, min_value=0.1, step=0.5)
    pixel_size = st.sidebar.number_input("Pixel size (mm/pixel)", value=0.1, min_value=0.01, step=0.01,
                                          help="Calibrate by measuring a known dimension in the image")
    
    tolerances = {'max_length_mm': max_length, 'max_area_mm2': max_area}

    st.sidebar.write("---")
    st.sidebar.write(f"**Active Limits:**")
    st.sidebar.write(f"Max Length: {max_length} mm")
    st.sidebar.write(f"Max Area: {max_area} mm²")

    # ---- MAIN: Upload + Detection ----
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📸 Upload Product Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if uploaded_file:
            st.write("### 🔎 Inspection Result")
            
            # Step 1: Classification
            prediction, confidence = classify(image, classifier)
            is_defective = 0 if prediction == "good" else 1

            if is_defective:
                st.error(f"⚠️ DEFECT DETECTED: **{prediction.upper()}**")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                st.write(f"**Defect Type:** {prediction}")
            else:
                st.success(f"✅ PART OK — No defect detected")
                st.metric("Confidence", f"{confidence*100:.1f}%")

    # ---- SEGMENTATION + MEASUREMENT (only if defective + model exists) ----
    if uploaded_file and is_defective and seg_model_exists:
        st.write("---")
        st.write("### 🔬 Defect Segmentation & Measurement")
        
        # Run segmentation
        mask = segment_defect(image, seg_model)
        
        # Resize image to match mask for overlay
        img_resized = np.array(image.resize((256, 256)))
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_resized)
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title("Defect Mask", fontsize=14)
        axes[1].axis('off')
        
        overlay = img_resized.copy()
        overlay[mask > 0.5] = [255, 0, 0]
        axes[2].imshow(overlay)
        axes[2].set_title("Defect Overlay", fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Measure defects
        measurements = measure_defect(mask, pixel_size)
        
        if measurements:
            results = check_tolerance(measurements, tolerances)
            
            st.write("### 📏 Defect Measurements vs Tolerance")
            
            all_passed = True
            total_area = 0
            max_len = 0
            
            for idx, r in enumerate(results):
                m = r['measurement']
                total_area += m['area_mm2']
                max_len = max(max_len, m['max_length_mm'])
                
                if r['passed']:
                    st.success(f"**Defect Region {idx+1}:** Length: {m['max_length_mm']}mm | Area: {m['area_mm2']}mm² → ✅ WITHIN TOLERANCE")
                else:
                    all_passed = False
                    st.error(f"**Defect Region {idx+1}:** Length: {m['max_length_mm']}mm | Area: {m['area_mm2']}mm² → ❌ EXCEEDS TOLERANCE")
                    for reason in r['reasons']:
                        st.write(f"  ⚠️ {reason}")
            
            # Final verdict
            st.write("---")
            if all_passed:
                st.warning(f"📋 **VERDICT: ACCEPT WITH DEVIATION** — Defect present but within tolerance limits")
            else:
                st.error(f"📋 **VERDICT: REJECT** — Defect exceeds batch tolerance limits")
            
            # Log with measurements
            log_inspection(prediction, confidence, is_defective, total_area, max_len, int(all_passed))
        else:
            st.info("Segmentation found no measurable defect regions")
            log_inspection(prediction, confidence, is_defective)
    
    elif uploaded_file and not is_defective:
        log_inspection(prediction, confidence, is_defective)
    
    elif uploaded_file and is_defective and not seg_model_exists:
        st.info("ℹ️ Segmentation model not found. Run train_segmentation.py first for measurement features.")
        log_inspection(prediction, confidence, is_defective)

    # ---- DASHBOARD ----
    st.write("---")
    st.write("### 📊 Inspection Dashboard")

    conn = sqlite3.connect('defect_logs.db')
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    
    if total > 0:
        defective = conn.execute("SELECT COUNT(*) FROM inspections WHERE is_defective = 1").fetchone()[0]
        good = total - defective
        
        # Try to get tolerance failures (column might not exist in old DB)
        try:
            rejected = conn.execute("SELECT COUNT(*) FROM inspections WHERE within_tolerance = 0").fetchone()[0]
        except:
            rejected = 0

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Total Inspections", total)
        dcol2.metric("Good Parts", good)
        dcol3.metric("Defective", defective)
        dcol4.metric("Rejected (Over Tolerance)", rejected)

        # Defect type breakdown
        rows = conn.execute("SELECT prediction, COUNT(*) FROM inspections WHERE is_defective = 1 GROUP BY prediction").fetchall()
        if rows:
            st.write("**Defect Type Breakdown:**")
            for row in rows:
                st.write(f"- **{row[0]}**: {row[1]} occurrences")

        # Recent inspections
        st.write("### 📋 Recent Inspections")
        recent = conn.execute(
            "SELECT prediction, confidence, is_defective, defect_length_mm, defect_area_mm2, within_tolerance, timestamp FROM inspections ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        
        for row in recent:
            status = "❌ DEFECT" if row[2] else "✅ GOOD"
            tol_status = ""
            if row[2]:  # if defective
                tol_status = " | ✅ Within Tol" if row[5] else " | ❌ Over Tol"
                tol_status += f" | L:{row[3]}mm A:{row[4]}mm²" if row[3] else ""
            st.write(f"{status} | {row[0]} | {row[1]*100:.1f}% conf{tol_status} | {row[6]}")

    conn.close()

if __name__ == "__main__":
    main()