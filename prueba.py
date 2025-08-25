import pyttsx3

engine = pyttsx3.init()

# Ajustar velocidad de la voz
rate = engine.getProperty("rate")
print(f"Velocidad actual: {rate}")  # normalmente ~200
engine.setProperty("rate", 140)     # ponlo más bajo para hacerlo más lento

engine.say("como estas")
engine.runAndWait()