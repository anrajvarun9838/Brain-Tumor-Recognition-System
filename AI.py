from flask import Blueprint, request, jsonify
from groq import Groq

ai_bp = Blueprint('ai_bp', __name__)
client = Groq(api_key="add your groq api key here")

medical_keywords = [
    "tumor", "mri", "brain", "glioma", "meningioma", "pituitary", "cancer", 
    "symptom", "treatment", "scan", "confidence", "result", "detected",
    "detection", "health", "medical","tumar","about",
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(keyword in query for keyword in medical_keywords)

@ai_bp.route('/ai/info', methods=['POST'])
def ai_info():
    data = request.get_json()
    user_query = data.get("query", "").strip()
    result_text = data.get("result", "").strip()
    confidence = data.get("confidence", "").strip()

    # AUTO QUERY 
    auto_query = ""
    if result_text and not user_query:
        auto_query = (
            f"MRI detected: {result_text} with {confidence}% confidence. "
            f"Explain what this means, possible symptoms, seriousness level, "
            f"and general guidance in simple language for a patient."
        )
    final_query = user_query if user_query else auto_query
    if not user_query and result_text:
        prompt = (
            f"The MRI analysis detected: {result_text} with a confidence of {confidence}. "
            f"Explain what this means in simple language for a non-medical person. "
            f"Provide key points, possible symptoms, and general advice (no specific medical diagnosis)."
        )
    elif user_query and not is_medical_query(user_query):
        return jsonify({
            "reply": (
                "I am specialized in MRI brain tumor analysis. "
                "Please ask a question related to MRI, brain health, or tumor detection."
            )
        })
    else:
        prompt = (
            f"The MRI system detected: {result_text} (confidence: {confidence}). "
            f"User asked: {final_query}. "
            f"Answer in an informative but simple tone, with short and clear explanation."
        )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI medical assistant specialized in MRI brain tumor analysis."},
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )

        reply = response.choices[0].message.content.strip()

        return jsonify({
            "reply": reply,
            "auto_used": auto_query  
        })

    except Exception as e:
        return jsonify({"reply": f"Error contacting AI: {str(e)}"})
