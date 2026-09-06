import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from dotenv import load_dotenv
from root_cause import get_root_cause_analysis

load_dotenv()

# ============================================
# Load classifier
# ============================================
class_names = ["good", "bent", "color", "flip", "scratch"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('best_defect_model_metalnut.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================
# Get 10 defective images (2 per defect type)
# ============================================
dataset_path = r"D:\envs\VSCODE_AI_Bootcamp\My_Projects\ManuVision AI\MVTec AD datase\metal_nut\test"

test_images = []
for defect_type in sorted(os.listdir(dataset_path)):
    if defect_type == "good":
        continue
    defect_dir = os.path.join(dataset_path, defect_type)
    imgs = sorted(os.listdir(defect_dir))[:2]  # take 2 per defect type
    for img_name in imgs:
        test_images.append({
            "path": os.path.join(defect_dir, img_name),
            "defect_type": defect_type
        })
print(f"Testing {len(test_images)} defective images\n")

# ============================================
# Run each through pipeline
# ============================================
results = []
for i, item in enumerate(test_images):
    print(f"{'='*60}")
    print(f"Image {i+1}: {item['defect_type']} — {os.path.basename(item['path'])}")
    print(f"{'='*60}")
    
    # Classify
    img = Image.open(item['path']).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    prediction = class_names[pred.item()]
    confidence = conf.item()
    
    print(f"Predicted: {prediction} ({confidence*100:.1f}%)")
    print(f"Actual: {item['defect_type']}")
    print(f"Correct: {'✅' if prediction == item['defect_type'] else '❌'}")
    
    # Get root cause analysis
    print(f"\nRoot Cause Analysis:")
    print(f"-" * 40)
    
    try:
        analysis = get_root_cause_analysis(
            defect_type=prediction,
            measurements=[{"area_mm2": 5.0, "max_length_mm": 2.5}],
            confidence=confidence
        )
        print(analysis)
    except Exception as e:
        analysis = f"Error: {str(e)}"
        print(analysis)
    
    results.append({
        "image": os.path.basename(item['path']),
        "actual": item['defect_type'],
        "predicted": prediction,
        "confidence": confidence,
        "correct": prediction == item['defect_type'],
        "root_cause": analysis
    })
    
    print(f"\n")

# ============================================
# Summary
# ============================================
print(f"{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")

correct = sum(1 for r in results if r['correct'])
print(f"Classification accuracy: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")

# Save results to file
with open("rootcause_evaluation.md", "w", encoding="utf-8") as f:
    f.write("# Root Cause Analysis Evaluation\n\n")
    f.write(f"Total images tested: {len(results)}\n")
    f.write(f"Classification accuracy: {correct}/{len(results)} ({100*correct/len(results):.0f}%)\n\n")
    
    for i, r in enumerate(results):
        f.write(f"## Image {i+1}: {r['actual']}\n\n")
        f.write(f"- **Predicted:** {r['predicted']} ({r['confidence']*100:.1f}%)\n")
        f.write(f"- **Correct classification:** {'Yes' if r['correct'] else 'No'}\n")
        f.write(f"- **Root Cause Analysis:**\n\n{r['root_cause']}\n\n")
        f.write(f"- **Expert Assessment:** [YOUR RATING: Correct / Partially Correct / Incorrect]\n")
        f.write(f"- **Expert Notes:** [YOUR NOTES]\n\n")
        f.write(f"---\n\n")

print(f"\nResults saved to rootcause_evaluation.md")
print(f"Open it and fill in YOUR expert assessment for each root cause!")