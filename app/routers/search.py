from fastapi import APIRouter
from pydantic import BaseModel
from app.services.vector_service import vector_db

router = APIRouter(prefix="/search", tags=["Semantic Search"])

class SearchRequest(BaseModel):
    texte_actuel: str
    categorie: str = None
    gravite_max: str = None

@router.post("/")
async def search_similar_claims(request: SearchRequest):
    # Interrogation de la base vectorielle FAISS
    # On renvoie les 3 plaintes passées les plus ressemblantes pour souffler une solution
    similar_cases = vector_db.search_similar(
        query=request.texte_actuel,
        top_k=3,
        category_filter=request.categorie
    )
    
    return {
        "message": "Recherche sémantique réussie",
        "resultats_trouves": len(similar_cases),
        "similar_claims": similar_cases
    }
