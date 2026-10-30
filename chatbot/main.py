import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
import os

# 1. API Configuration
API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)

try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    SELECTED_MODEL = available_models[0] # بياخد أول واحد شغال (غالباً 1.5-flash)
except:
    SELECTED_MODEL = "models/gemini-1.5-flash"

model = genai.GenerativeModel(SELECTED_MODEL)

SYSTEM_INSTRUCTION = (
    "أنت المساعد الذكي لبرنامج LiverGuardian. "
    "قواعد صارمة للرد: "
    "1. الرد بالعامية المصرية فقط. "
    "2. ممنوع استخدام رموز غريبة، اجعل النص بسيطاً. "
    "3. استخدم نظام النقط (Bullet points) لتنظيم النص وجعله مريحاً للعين. "
    "4. ابدأ بالترحيب، ثم الإجابة، ثم النصيحة بزيارة الطبيب. "
    "5. لا تطل في الكلام، كن مختصراً ومفيداً."
)

app = FastAPI(title="LiverGuardian AI Service")

class ChatRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_liver_guardian(request: ChatRequest):
    try:
        prompt = f"{SYSTEM_INSTRUCTION}\n\nسؤال المستخدم: {request.message}"
        
        response = model.generate_content(prompt)
        
        final_text = response.text.strip()
        
        return {
            "status": "success",
            "answer": final_text
        }
    except Exception as e:
        return {
            "status": "error",
            "error_details": str(e)
        }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)    
# To run the server:
# uvicorn main:app --reload
