
import requests
import re

url = 'http://127.0.0.1:5000/predict'
# Data for a "Malignant" case (high values)
data = {
    'clump_thickness': '10',
    'uniform_cell_size': '10',
    'uniform_cell_shape': '10',
    'marginal_adhesion': '10',
    'single_epithelial_size': '10',
    'bare_nuclei': '10',
    'bland_chromatin': '10',
    'normal_nucleoli': '10',
    'mitoses': '10'
}

try:
    response = requests.post(url, data=data)
    if response.status_code == 200:
        content = response.text
        # Look for confidence in simple text or the JavaScript part
        # "Confidence: <span ...>95.5%</span>" or just look for the number
        match = re.search(r'Confidence:\s*([\d\.]+)%', content)
        if match:
            print(f"Components Found: {match.group(0)}")
            print(f"Confidence Score: {match.group(1)}")
        else:
            # Fallback search in case of different formatting
            print("Could not find structured confidence string. Dumping snippet:")
            print(content[:500]) # First 500 chars might not have it, usually it's further down
            
    else:
        print("Status Code:", response.status_code)
except Exception as e:
    print("Error:", e)
