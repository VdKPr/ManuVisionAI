import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import json

load_dotenv()

# ============================================
# LOAD MODELS (reuse from ManuVision AI)
# ============================================
class_names = ["good", "bent", "color", "flip", "scratch"]

# UNet definition (same as before)
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
        self.enc1=DoubleConv(3,64); self.enc2=DoubleConv(64,128)
        self.enc3=DoubleConv(128,256); self.enc4=DoubleConv(256,512)
        self.bottleneck=DoubleConv(512,1024)
        self.up4=nn.ConvTranspose2d(1024,512,2,stride=2); self.dec4=DoubleConv(1024,512)
        self.up3=nn.ConvTranspose2d(512,256,2,stride=2); self.dec3=DoubleConv(512,256)
        self.up2=nn.ConvTranspose2d(256,128,2,stride=2); self.dec2=DoubleConv(256,128)
        self.up1=nn.ConvTranspose2d(128,64,2,stride=2); self.dec1=DoubleConv(128,64)
        self.final=nn.Conv2d(64,1,1); self.pool=nn.MaxPool2d(2)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return torch.sigmoid(self.final(d1))

# Load models
classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, len(class_names))
classifier.load_state_dict(torch.load('best_defect_model.pth', map_location='cpu'))
classifier.eval()

seg_model = UNet()
seg_model.load_state_dict(torch.load('best_segmentation_model.pth', map_location='cpu'))
seg_model.eval()

classify_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
segment_tf = transforms.Compose([
    transforms.Resize((256,256)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================================
# TOOLS — functions the agent can call
# ============================================
def tool_classify(image_path: str) -> dict:
    """Classify defect type in image"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = classify_tf(img).unsqueeze(0)
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return {
        "tool": "classifier",
        "prediction": class_names[pred.item()],
        "confidence": round(conf.item() * 100, 1),
        "is_defective": class_names[pred.item()] != "good"
    }

def tool_segment(image_path: str) -> dict:
    """Segment defect region and measure it"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = segment_tf(img).unsqueeze(0)
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
    
    labeled, num = ndimage.label(mask)
    measurements = []
    pixel_size = 0.1
    for i in range(1, num+1):
        region = (labeled==i)
        area = region.sum() * (pixel_size**2)
        coords = np.where(region)
        if len(coords[0]) == 0: continue
        h = (coords[0].max()-coords[0].min()) * pixel_size
        w = (coords[1].max()-coords[1].min()) * pixel_size
        measurements.append({
            "area_mm2": round(area, 2),
            "max_length_mm": round(max(h,w), 2)
        })
    
    return {
        "tool": "segmentation",
        "num_defect_regions": len(measurements),
        "measurements": measurements,
        "total_defect_area_mm2": round(sum(m["area_mm2"] for m in measurements), 2)
    }

def tool_tolerance_check(measurements: list, max_length: float = 2.0, max_area: float = 5.0) -> dict:
    """Check measurements against tolerance limits"""
    results = []
    all_passed = True
    for m in measurements:
        passed = True
        reasons = []
        if m["max_length_mm"] > max_length:
            passed = False
            reasons.append(f"Length {m['max_length_mm']}mm exceeds {max_length}mm")
        if m["area_mm2"] > max_area:
            passed = False
            reasons.append(f"Area {m['area_mm2']}mm² exceeds {max_area}mm²")
        if not passed:
            all_passed = False
        results.append({"passed": passed, "reasons": reasons})
    
    return {
        "tool": "tolerance_check",
        "verdict": "PASS" if all_passed else "REJECT",
        "details": results
    }

def tool_root_cause(defect_type: str, measurements: list, confidence: float) -> str:
    """Generate root cause analysis using LLM"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    meas_text = ", ".join([
        f"Region {i+1}: length={m['max_length_mm']}mm, area={m['area_mm2']}mm²"
        for i, m in enumerate(measurements)
    ]) if measurements else "No measurements available"
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior manufacturing quality engineer.

Defect Type: {defect_type}
Confidence: {confidence}%
Measurements: {measurements}

Provide a concise root cause analysis:
1. Most probable cause (one paragraph)
2. Top 3 corrective actions (one line each)
3. Severity: Low/Medium/High/Critical
4. Recommended disposition: Use As-Is / Rework / Scrap"""
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "defect_type": defect_type,
        "confidence": str(confidence),
        "measurements": meas_text
    })
    return response.content

# ============================================
# AGENT STATE
# ============================================
class AgentState(TypedDict):
    image_path: str
    classification: dict
    segmentation: dict
    tolerance: dict
    root_cause: str
    final_report: str
    step_log: list

# ============================================
# AGENT NODES — each node is a step the agent takes
# ============================================
def classify_node(state: AgentState) -> AgentState:
    print("🔍 Agent: Running classification...")
    result = tool_classify(state["image_path"])
    state["classification"] = result
    state["step_log"].append(f"Step 1: Classified as '{result['prediction']}' ({result['confidence']}% confidence)")
    return state

def should_segment(state: AgentState) -> str:
    """Agent DECIDES: should I segment or skip?"""
    if state["classification"]["is_defective"]:
        print("🧠 Agent: Defect found. I need to segment to measure it.")
        return "segment"
    else:
        print("🧠 Agent: Part is good. No segmentation needed.")
        return "report"

def segment_node(state: AgentState) -> AgentState:
    print("🔬 Agent: Running segmentation + measurement...")
    result = tool_segment(state["image_path"])
    state["segmentation"] = result
    state["step_log"].append(f"Step 2: Found {result['num_defect_regions']} defect regions, total area: {result['total_defect_area_mm2']}mm²")
    return state

def tolerance_node(state: AgentState) -> AgentState:
    print("📏 Agent: Checking tolerances...")
    measurements = state["segmentation"]["measurements"]
    result = tool_tolerance_check(measurements)
    state["tolerance"] = result
    state["step_log"].append(f"Step 3: Tolerance verdict: {result['verdict']}")
    return state

def root_cause_node(state: AgentState) -> AgentState:
    print("🧠 Agent: Generating root cause analysis...")
    analysis = tool_root_cause(
        state["classification"]["prediction"],
        state["segmentation"]["measurements"],
        state["classification"]["confidence"]
    )
    state["root_cause"] = analysis
    state["step_log"].append("Step 4: Root cause analysis generated")
    return state

def report_node(state: AgentState) -> AgentState:
    print("📋 Agent: Generating final report...")
    
    classification = state["classification"]
    
    if not classification["is_defective"]:
        report = f"""
═══════════════════════════════════════════
  MANUVISION AI — INSPECTION REPORT
═══════════════════════════════════════════
  
  Image: {state['image_path']}
  
  RESULT: ✅ PASS — No defect detected
  Confidence: {classification['confidence']}%
  
  Agent Decision Log:
  {chr(10).join(state['step_log'])}
═══════════════════════════════════════════"""
    else:
        segmentation = state.get("segmentation", {})
        tolerance = state.get("tolerance", {})
        root_cause = state.get("root_cause", "N/A")
        
        report = f"""
═══════════════════════════════════════════
  MANUVISION AI — INSPECTION REPORT
═══════════════════════════════════════════

  Image: {state['image_path']}

  CLASSIFICATION:
    Defect Type: {classification['prediction']}
    Confidence: {classification['confidence']}%
  
  SEGMENTATION:
    Defect Regions: {segmentation.get('num_defect_regions', 'N/A')}
    Total Defect Area: {segmentation.get('total_defect_area_mm2', 'N/A')} mm²
    Measurements: {json.dumps(segmentation.get('measurements', []), indent=4)}
  
  TOLERANCE CHECK:
    Verdict: {tolerance.get('verdict', 'N/A')}
  
  ROOT CAUSE ANALYSIS:
    {root_cause}
  
  AGENT DECISION LOG:
    {chr(10).join('    ' + s for s in state['step_log'])}
    
═══════════════════════════════════════════"""
    
    state["final_report"] = report
    return state

# ============================================
# BUILD THE AGENT GRAPH
# ============================================
def build_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("segment", segment_node)
    workflow.add_node("tolerance", tolerance_node)
    workflow.add_node("root_cause", root_cause_node)
    workflow.add_node("report", report_node)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Agent DECIDES whether to segment based on classification
    workflow.add_conditional_edges(
        "classify",
        should_segment,
        {
            "segment": "segment",
            "report": "report"
        }
    )
    
    # After segmentation → check tolerance → root cause → report
    workflow.add_edge("segment", "tolerance")
    workflow.add_edge("tolerance", "root_cause")
    workflow.add_edge("root_cause", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()

# ============================================
# RUN THE AGENT
# ============================================
def run_inspection(image_path: str):
    agent = build_agent()
    
    initial_state = {
        "image_path": image_path,
        "classification": {},
        "segmentation": {},
        "tolerance": {},
        "root_cause": "",
        "final_report": "",
        "step_log": []
    }
    
    print(f"\n🤖 ManuVision Agent starting inspection: {image_path}\n")
    
    result = agent.invoke(initial_state)
    
    print(result["final_report"])
    return result

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image — change this to your MVTec image path
        #image_path = r"D:/envs/VSCODE_AI_Bootcamp/My_Projects/ManuVision AI/MVTec AD dataset/metal_nut/test/bent/000.png"
        # Change this line at the bottom of agent.py:

        #image_path = r"D:/envs/VSCODE_AI_Bootcamp/My_Projects/ManuVision AI/MVTec AD dataset/metal_nut/test/bent/000.png"
        image_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut\test\bent\001.png"
    result = run_inspection(image_path)
