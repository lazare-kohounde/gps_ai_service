from fastapi import APIRouter
from pydantic import BaseModel
from app.services.nlp_service import analyze_sentiment_and_urgency, generate_short_summary

router = APIRouter(prefix="/analyze", tags=["NLP Analysis"])

class TextRequest(BaseModel):
    texte: str

@router.post("/")
async def analyze_text(request: TextRequest):
    # Appel de la logique du service NLP
    texte = request.texte
    gravity, sentiment, mots_cles = analyze_sentiment_and_urgency(texte)
    resume = generate_short_summary(texte)
    
    return {
        "urgence": gravity, 
        "sentiment": sentiment,
        "mots_cles_detectes": mots_cles,
        "resume": resume
    }
