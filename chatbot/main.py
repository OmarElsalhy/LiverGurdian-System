import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
import os

# 1. API Configuration
API_KEY = "AIzaSyA4tReP15S63eOxmnd5Ik-z2HsWOduPV_0"
genai.configure(api_key=API_KEY)

# جلب الموديل المتاح أوتوماتيكياً عشان نتفادى الـ 404
try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    SELECTED_MODEL = available_models[0] # بياخد أول واحد شغال (غالباً 1.5-flash)
except:
    SELECTED_MODEL = "models/gemini-1.5-flash"

model = genai.GenerativeModel(SELECTED_MODEL)

# 2. دستور الرد (هنا ركزنا على الترتيب ونظافة الكلام)
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
        # دمج الدستور مع السؤال
        prompt = f"{SYSTEM_INSTRUCTION}\n\nسؤال المستخدم: {request.message}"
        
        response = model.generate_content(prompt)
        
        # تنظيف النص الناتج من أي مسافات أو رموز زائدة
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