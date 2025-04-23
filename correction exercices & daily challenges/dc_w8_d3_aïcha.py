from transformers import pipeline
import gradio as gr

# Charger le modèle BlenderBot
chatbot = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

# Contexte global
context = ""

# Fonction principale du chatbot
def mini_chatbot(user_message, history):
    global context

    # Créer le texte avec historique
    input_text = context + f"\nUser: {user_message}\nBot: "
    
    # DEBUG : afficher le texte envoyé au modèle
    print("\n📝 Input envoyé au modèle :")
    print(input_text)
    print("-" * 60)

    try:
        # Génération de réponse
        response = chatbot(input_text, max_length=200, do_sample=True)[0]["generated_text"]

        # DEBUG : afficher la réponse brute
        print("🤖 Réponse générée :", response)

        # Mettre à jour le contexte
        context += f"\nUser: {user_message}\nBot: {response}"

    except Exception as e:
        print("❌ Erreur pendant la génération :", e)
        response = "Oops, something went wrong. Let's try again."

        # Réinitialiser le contexte en cas de plantage
        context = ""

    return response

# Fonction pour réinitialiser le contexte via Gradio
def reset_chat():
    global context
    context = ""
    print("🔄 Contexte réinitialisé.")
    return None

# Interface Gradio
demo_chatbot = gr.ChatInterface(
    fn=mini_chatbot,
    title="BlenderBot Chatbot 🤖",
    description="Chat with a BlenderBot-based conversational AI from Facebook."
)

# Attacher la fonction de reset au bouton de nettoyage
demo_chatbot.chatbot.clear_fn = reset_chat

# Lancer l’application
if __name__ == "__main__":
    demo_chatbot.launch()
