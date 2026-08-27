import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# You already know this from DocVision AI — same pattern
def get_root_cause_analysis(defect_type, measurements, confidence):
    """
    Given a detected defect, use LLM to generate root cause analysis
    with manufacturing-specific recommendations.
    """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior manufacturing quality engineer with 20 years of experience 
in metal component manufacturing (stamping, casting, machining, heat treatment).

A defect has been detected by our AI inspection system on a metal nut:

**Defect Type:** {defect_type}
**Detection Confidence:** {confidence}%
**Defect Measurements:** {measurements}

Provide a detailed root cause analysis in the following format:

## Root Cause Analysis

### Most Probable Cause
(Explain the most likely manufacturing process issue that caused this defect)

### Contributing Factors
(List 2-3 other factors that may have contributed)

### Recommended Corrective Actions
1. (Immediate action to take)
2. (Process adjustment)
3. (Preventive measure)

### Quality Impact Assessment
- Severity: (Low/Medium/High/Critical)
- Likelihood of recurrence: (Low/Medium/High)
- Recommended disposition: (Use As-Is / Rework / Scrap)

### Process Parameters to Check
(List specific machine settings or parameters that should be verified)

Be specific to metal nut manufacturing. Reference actual process parameters 
like press force (kN), temperature (°C), feed rate (mm/min) where applicable.
Do NOT use generic advice — be specific and actionable."""
    )
    
    chain = prompt | llm
    
    # Format measurements for the prompt
    if measurements:
        meas_text = ", ".join([
            f"Region {i+1}: length={m['max_length_mm']}mm, area={m['area_mm2']}mm²" 
            for i, m in enumerate(measurements)
        ])
    else:
        meas_text = "No detailed measurements available"
    
    response = chain.invoke({
        "defect_type": defect_type,
        "confidence": f"{confidence*100:.1f}",
        "measurements": meas_text
    })
    
    return response.content