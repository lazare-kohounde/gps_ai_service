import ollama
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèle Ollama à utiliser par défaut (ex: llama3, mistral, ou phi3 pour plus de légèreté)
# Assurez-vous d'avoir téléchargé le modèle localement avec : ollama run llama3.2:1b
DEFAULT_MODEL = "llama3.2:1b"

def generate_solution_from_history(current_complaint_text: str, similar_historic_solutions: list) -> str:
    """
    Génère une proposition de solution unique et formatée en se basant sur la plainte actuelle
    et un échantillon des meilleures solutions passées.
    """
    if not similar_historic_solutions:
        return "Aucune solution historique trouvée pour formuler une recommandation."

    # 1. Construction du contexte (Les solutions passées)
    context_text = "\n".join([f"- Historique {i+1} : {sol}" for i, sol in enumerate(similar_historic_solutions)])

    # 2. Construction du Prompt Système (Les instructions strictes pour l'IA)
    system_prompt = """
    Tu es un assistant expert pour le service client de GPR (Gestion des Réclamations).
    Ton rôle est d'analyser une plainte d'un client actuel, de regarder comment des problèmes similaires ont été résolus dans le passé, 
    et de rédiger UNE SEULE proposition de réponse/solution claire, professionnelle et prête à être envoyée au client.
    
    Règles strictes :
    - Reste courtois et professionnel (vouvoiement).
    - Ne copie pas textuellement les historiques, inspire-t-en pour créer une réponse sur-mesure.
    - Ne mentionne pas que tu es une IA ni que tu t'inspires d'historiques.
    - Va droit au but. Propose des solutions concrètes (ex: annulation de frais, vérification technique en cours, etc.) basées sur le contexte.
    """

    # 3. Construction de la requête utilisateur (La tâche)
    user_prompt = f"""
    Plainte actuelle du client :
    "{current_complaint_text}"

    Voici comment des plaintes similaires ont été résolues par nos agents par le passé :
    {context_text}

    Rédige maintenant la solution idéale à proposer pour cette plainte actuelle.
    """

    try:
        logger.info(f"Appel au modèle LLM local '{DEFAULT_MODEL}' via Ollama...")
        
        response = ollama.chat(model=DEFAULT_MODEL, messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ])
        
        generated_text = response['message']['content']
        logger.info("Génération LLM réussie.")
        return generated_text.strip()

    except Exception as e:
        logger.error(f"Erreur lors de l'appel à Ollama : {e}")
        return "Erreur : Impossible de générer une solution. Veuillez vérifier que Ollama est bien lancé en arrière-plan."

