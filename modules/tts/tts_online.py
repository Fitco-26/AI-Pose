from gtts import gTTS
import os
import tempfile
import playsound

class TTSOnline:
    def __init__(self, lang="en"):
        """
        Initialize the TTS system.
        :param lang: Language code (default: 'en')
        """
        self.lang = lang

    def speak(self, text: str):
        """
        Convert text to speech and play it.
        :param text: The text to be spoken
        """
        try:
            # Convert text to speech
            tts = gTTS(text=text, lang=self.lang)

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                tmp_path = tmp_file.name

            # Play the file
            playsound.playsound(tmp_path, True)

            # Clean up temp file
            os.remove(tmp_path)

        except Exception as e:
            print(f"[TTS ERROR] {e}")
