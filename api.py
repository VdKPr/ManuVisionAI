from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing
import sqlite3
import io
from datetime import datetime
from dotenv import load_dotenv
from root_cause import get_root_cause_analysis

load_dotenv()

# ============================================
# MODELS + CONFIG (same as app.py)
# ============================================
class_names = ["good", "bent", "color", "flip", "scratch"]

# U-Net definition (same as app.py)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 64); self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256); self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, 1); self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return torch.sigmoid(self.final(d1))

# Load models once at startup
classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, len(class_names))
classifier.load_state_dict(torch.load('best_defect_model.pth', map_location='cpu'))
classifier.eval()

seg_model = UNet()
seg_model.load_state_dict(torch.load('best_segmentation_model.pth', map_location='cpu'))
seg_model.eval()

# Transforms
classify_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
segment_tf = transforms.Compose([
    transforms.Resize((256,256)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================================
# DATABASE
# ============================================
def init_db():
    conn = sqlite3.connect('defect_logs.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS inspections
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     prediction TEXT, confidence REAL, is_defective INTEGER,
                     defect_area_mm2 REAL, defect_length_mm REAL,
                     within_tolerance INTEGER,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="ManuVision AI API",
    description="AI-Powered Manufacturing Defect Detection, Segmentation, Measurement & Root Cause Analysis",
    version="1.0.0"
)

# ============================================
# HELPER FUNCTIONS
# ============================================
def classify(image):
    img_tensor = classify_tf(image).unsqueeze(0)
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item()

def segment(image):
    img_tensor = segment_tf(image).unsqueeze(0)
    with torch.no_grad():
        mask = seg_model(img_tensor)
    mask = mask.squeeze().numpy()
    mask = (mask > 0.25).astype(np.uint8)
    mask = binary_opening(mask, structure=np.ones((5,5))).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((3,3))).astype(np.uint8)
    labeled, num = ndimage.label(mask)
    for i in range(1, num+1):
        if (labeled==i).sum() < 50:
            mask[labeled==i] = 0
    return mask

def measure(mask, pixel_size_mm=0.1):
    labeled, num = ndimage.label(mask)
    measurements = []
    for i in range(1, num+1):
        region = (labeled==i)
        area_px = region.sum()
        coords = np.where(region)
        if len(coords[0]) == 0: continue
        h = (coords[0].max()-coords[0].min()) * pixel_size_mm
        w = (coords[1].max()-coords[1].min()) * pixel_size_mm
        measurements.append({
            'area_mm2': round(area_px*(pixel_size_mm**2), 2),
            'max_length_mm': round(max(h,w), 2)
        })
    return measurements

def log_to_db(prediction, confidence, is_defective, area=0, length=0, within_tol=1):
    conn = sqlite3.connect('defect_logs.db')
    conn.execute(
        "INSERT INTO inspections (prediction,confidence,is_defective,defect_area_mm2,defect_length_mm,within_tolerance) VALUES (?,?,?,?,?,?)",
        (prediction, confidence, is_defective, area, length, within_tol)
    )
    conn.commit(); conn.close()

# ============================================
# API ENDPOINTS
# ============================================

class ToleranceParams(BaseModel):
    max_length_mm: float = 2.0
    max_area_mm2: float = 5.0
    pixel_size_mm: float = 0.1

@app.get("/")
def root():
    return {"message": "ManuVision AI API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/inspect")
async def inspect_part(
    file: UploadFile = File(...),
    max_length_mm: float = 2.0,
    max_area_mm2: float = 5.0,
    pixel_size_mm: float = 0.1
):
    """Full inspection pipeline: classify → segment → measure → tolerance → root cause"""
    
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Step 1: Classification
    prediction, confidence = classify(image)
    is_defective = 0 if prediction == "good" else 1
    
    result = {
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
        "is_defective": bool(is_defective),
        "measurements": [],
        "tolerance_check": [],
        "verdict": "PASS",
        "root_cause_analysis": None
    }
    
    if is_defective:
        # Step 2: Segmentation
        mask = segment(image)
        
        # Step 3: Measurement
        measurements = measure(mask, pixel_size_mm)
        result["measurements"] = measurements
        
        # Step 4: Tolerance check
        all_passed = True
        tolerance_results = []
        total_area = 0
        max_len = 0
        
        for m in measurements:
            passed = True
            reasons = []
            total_area += m['area_mm2']
            max_len = max(max_len, m['max_length_mm'])
            
            if m['max_length_mm'] > max_length_mm:
                passed = False
                reasons.append(f"Length {m['max_length_mm']}mm exceeds {max_length_mm}mm")
            if m['area_mm2'] > max_area_mm2:
                passed = False
                reasons.append(f"Area {m['area_mm2']}mm² exceeds {max_area_mm2}mm²")
            
            if not passed: all_passed = False
            tolerance_results.append({"passed": passed, "reasons": reasons})
        
        result["tolerance_check"] = tolerance_results
        result["verdict"] = "PASS — within tolerance" if all_passed else "REJECT — exceeds tolerance"
        
        # Step 5: Root cause analysis
        try:
            analysis = get_root_cause_analysis(prediction, measurements, confidence)
            result["root_cause_analysis"] = analysis
        except Exception as e:
            result["root_cause_analysis"] = f"Error generating analysis: {str(e)}"
        
        # Log to database
        log_to_db(prediction, confidence, is_defective, total_area, max_len, int(all_passed))
    else:
        log_to_db(prediction, confidence, is_defective)
    
    return result

@app.get("/dashboard")
def get_dashboard():
    """Get inspection statistics and recent history"""
    conn = sqlite3.connect('defect_logs.db')
    
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    if total == 0:
        conn.close()
        return {"total": 0, "message": "No inspections yet"}
    
    defective = conn.execute("SELECT COUNT(*) FROM inspections WHERE is_defective=1").fetchone()[0]
    
    try:
        rejected = conn.execute("SELECT COUNT(*) FROM inspections WHERE within_tolerance=0").fetchone()[0]
    except:
        rejected = 0
    
    # Defect breakdown
    breakdown = conn.execute(
        "SELECT prediction, COUNT(*) FROM inspections WHERE is_defective=1 GROUP BY prediction"
    ).fetchall()
    
    # Recent inspections
    recent = conn.execute(
        "SELECT prediction, confidence, is_defective, defect_length_mm, defect_area_mm2, within_tolerance, timestamp FROM inspections ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()
    
    conn.close()
    
    return {
        "total_inspections": total,
        "good_parts": total - defective,
        "defective_parts": defective,
        "rejected_parts": rejected,
        "pass_rate": round((total-defective)/total*100, 1),
        "defect_breakdown": {row[0]: row[1] for row in breakdown},
        "recent_inspections": [
            {
                "prediction": r[0], "confidence": round(r[1]*100,1),
                "is_defective": bool(r[2]), "length_mm": r[3],
                "area_mm2": r[4], "within_tolerance": bool(r[5]),
                "timestamp": r[6]
            } for r in recent
        ]
    }

@app.get("/stats")
def get_stats():
    """Quick stats for monitoring"""
    conn = sqlite3.connect('defect_logs.db')
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    defective = conn.execute("SELECT COUNT(*) FROM inspections WHERE is_defective=1").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "defective": defective,
        "good": total - defective,
        "defect_rate": round(defective/total*100, 1) if total > 0 else 0
    }