import re
from textblob import TextBlob

# Dictionnaire de mots sensibles métier qui rehaussent automatiquement l'urgence
MOTS_SENSIBLES = [
    "fraude", "arnaque", "piratage", "vol", "police", "tribunal",
    "avocat", "presse", "bceao", "cima", "médiateur", "huissier",
    "inadmissible", "honte", "urgence", "plainte", "scandale"
]

def analyze_sentiment_and_urgency(text: str) -> tuple[str, str, list[str]]:
    """
    Analyse le texte pour déterminer le niveau de gravité (MINEUR, MOYEN, GRAVE),
    une tonalité globale, et les mots sensibles présents.
    """
    text_lower = text.lower()
    
    # 1. Détection des mots-clés métier
    detected_keywords = []
    for mot in MOTS_SENSIBLES:
        # Recherche du mot complet (éviter de matcher 'vol' dans 'volontaire')
        if re.search(rf"\b{mot}s?\b", text_lower):
            detected_keywords.append(mot)
    
    # 2. Analyse basique du sentiment via TextBlob (traduit en polarité de -1 à 1)
    # Note: TextBlob est basique en FR, mais suffisant pour détecter une forte négativité collée à la V1
    blob = TextBlob(text)
    # Pour un MVP, même avec TextBlob par défaut, on peut obtenir une proxy avec une lib dédiée FR
    # Mais ici on va se baser principalement sur la polarité perçue et les règles lexicales
    polarity = blob.sentiment.polarity
    
    sentiment = "neutre"
    if polarity < -0.3:
        sentiment = "tres_negatif"
    elif polarity < 0:
        sentiment = "negatif"
    
    # 3. Règles de calcul du niveau d'urgence (Alignées sur GravityLevel: MINEUR, MOYEN, GRAVE)
    # Règle 1: Si présence de mots "très sensibles" liés au juridique ou institution (BCEAO, justice)
    mots_critiques = ["fraude", "vol", "police", "avocat", "tribunal", "bceao", "cima", "médiateur", "huissier"]
    has_critical_word = any(mot in detected_keywords for mot in mots_critiques)
    
    if has_critical_word:
        gravity = "GRAVE"
    elif len(detected_keywords) >= 2 or polarity <= -0.5:
        # Plusieurs mots sensibles ou texte extrêmement négatif (beaucoup de majuscules/points d'exclamation)
        gravity = "MOYEN"
    else:
        gravity = "MINEUR"
        
    # Heuristique additionnelle : Beaucoup de ponctuation agressive (!!!)
    if text.count('!') >= 3:
        if gravity == "MINEUR": gravity = "MOYEN"
        
    return gravity, sentiment, detected_keywords

def generate_short_summary(text: str) -> str:
    """
    Génère un faux résumé IA (placeholder pour l'instant, ou phrase condensée).
    Pour un vrai NLP résumé : Utiliser le modèle BART-large-CNN ou similaire, 
    mais pour le MVP on va faire une extraction des premières phrases marquantes.
    """
    sentences = text.split('.')
    if len(sentences) > 2:
        # Prendre la première phrase et la dernière (souvent le problème + l'attente)
        return f"{sentences[0].strip()}. [...] {sentences[-2].strip()}."
    return text[:200] + ("..." if len(text)>200 else "")
