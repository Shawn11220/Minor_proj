import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from transformers import pipeline
import speech_recognition as sr
from scipy.stats import zscore
import pandas as pd
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class VoiceMentalHealthAssessment:
    def __init__(self):
        self.audio_data = None
        self.sample_rate = 16000
        self.emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        self.transcription = ""
        self.features = {}
        self.sentiment_scores = {}
        self.depression_score = 0
        self.anxiety_score = 0
        
    def record_audio(self, duration=20):
        """Record audio for the specified duration"""
        print(f"Recording for {duration} seconds...")
        self.audio_data = sd.rec(int(duration * self.sample_rate), 
                                samplerate=self.sample_rate, channels=1)
        sd.wait()
        print("Recording complete!")
        return self.audio_data
    
    def load_audio_file(self, file_path):
        """Load audio from a file"""
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return None
        
        try:
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
            print(f"Audio loaded from {file_path}")
            return self.audio_data
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
    
    def save_audio(self, file_path="recorded_audio.wav"):
        """Save the recorded audio to a file"""
        if self.audio_data is not None:
            sf.write(file_path, self.audio_data, self.sample_rate)
            print(f"Audio saved to {file_path}")
        else:
            print("No audio data to save")
    
    def remove_noise(self, noise_clip=None):
        """Remove background noise using spectral gating"""
        if self.audio_data is None:
            return None
            
        # If no noise profile provided, estimate from quiet parts
        if noise_clip is None:
            # Estimate noise from quietest sections
            frame_length = 2048
            hop_length = 512
            
            # Calculate signal power
            S = np.abs(librosa.stft(self.audio_data, n_fft=frame_length, hop_length=hop_length))
            power = S**2
            
            # Find noise floor (lowest 5% of power)
            noise_threshold = np.percentile(power, 5, axis=1)
            noise_mask = power < np.expand_dims(noise_threshold, 1)
            
            # Apply spectral gating
            S_cleaned = S * ~noise_mask
            self.audio_data = librosa.istft(S_cleaned, hop_length=hop_length)
        else:
            # Use provided noise profile
            noise_profile = librosa.stft(noise_clip)
            S_target = librosa.stft(self.audio_data)
            S_cleaned = S_target - np.mean(np.abs(noise_profile))
            self.audio_data = librosa.istft(S_cleaned)
            
        return self.audio_data

    def remove_silence(self, min_duration=0.1, threshold=0.01):
        """Remove silent segments from audio"""
        if self.audio_data is None:
            return None
            
        # Find non-silent intervals
        intervals = librosa.effects.split(self.audio_data, 
                                        top_db=20,
                                        frame_length=2048,
                                        hop_length=512)
        
        # Keep only segments longer than min_duration
        valid_intervals = [i for i in intervals 
                         if (i[1] - i[0]) / self.sample_rate >= min_duration]
        
        # Concatenate valid segments
        self.audio_data = np.concatenate([self.audio_data[start:end] 
                                        for start, end in valid_intervals])
        return self.audio_data

    def assess_audio_quality(self):
        """Assess the quality of the audio signal"""
        if self.audio_data is None:
            return None
            
        quality_metrics = {}
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(self.audio_data**2)
        noise_power = np.var(self.audio_data)
        quality_metrics['snr'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Peak-to-Average Power Ratio (PAPR)
        peak_power = np.max(np.abs(self.audio_data)**2)
        quality_metrics['papr'] = 10 * np.log10(peak_power / signal_power) if signal_power > 0 else 0
        
        # Clipping detection
        quality_metrics['clipping_ratio'] = np.mean(np.abs(self.audio_data) > 0.95)
        
        print("\nAudio Quality Metrics:")
        print(f"SNR: {quality_metrics['snr']:.2f} dB")
        print(f"PAPR: {quality_metrics['papr']:.2f} dB")
        print(f"Clipping: {quality_metrics['clipping_ratio']*100:.2f}%")
        
        return quality_metrics

    def preprocess_audio(self):
        """Perform audio preprocessing"""
        if self.audio_data is None:
            print("No audio data to process")
            return None
        
        # Ensure audio data is the right shape
        if len(self.audio_data.shape) > 1:
            self.audio_data = self.audio_data.flatten()
        
        # Assess initial audio quality
        print("Initial audio quality assessment:")
        self.assess_audio_quality()
        
        # Remove background noise
        print("Removing background noise...")
        self.remove_noise()
        
        # Remove silent segments
        print("Removing silence...")
        self.remove_silence()
        
        # Apply high-pass filter
        from scipy import signal
        b, a = signal.butter(5, 100/(self.sample_rate/2), 'highpass')
        self.audio_data = signal.filtfilt(b, a, self.audio_data)
        
        # Check if audio volume is low and needs amplification
        rms_level = np.sqrt(np.mean(self.audio_data**2))
        if rms_level < 0.1:
            print("Low audio volume detected, applying amplification...")
            self.audio_data *= min(2.0, 0.9/rms_level)
        
        # Final normalization
        self.audio_data = librosa.util.normalize(self.audio_data)
        
        # Final quality assessment
        print("\nFinal audio quality assessment:")
        self.assess_audio_quality()
        
        print("Audio preprocessing complete")
        return self.audio_data
    
    def process_audio(self):
        """Complete audio processing pipeline"""
        print("\nProcessing audio...")
        
        # Store original audio for transcription
        self.original_audio = np.copy(self.audio_data)
        
        # Transcribe audio
        print("\nTranscribing audio...")
        self.speech_to_text()
        
        # Continue with feature extraction and analysis
        self.extract_audio_features()
        # print(dir(self)) 
        self.analyze_text_sentiment()
        self.calculate_mental_health_scores()

    def speech_to_text(self):
        """Convert speech to text using speech recognition"""
        if self.audio_data is None:
            print("No audio data to transcribe")
            return ""
        
        # Save audio temporarily for speech recognition
        temp_file = "temp_audio.wav"
        sf.write(temp_file, self.audio_data, self.sample_rate)
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Convert speech to text
        try:
            with sr.AudioFile(temp_file) as source:
                audio = recognizer.record(source)
                self.transcription = recognizer.recognize_google(audio, language="en-IN")
                print(f"Transcription: {self.transcription}")
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            self.transcription = ""
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
            self.transcription = ""
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return self.transcription
    
    def extract_audio_features(self):
        """Extract audio features for emotion analysis"""
        if self.audio_data is None:
            print("No audio data to extract features from")
            return None
        
        # Extract fundamental features
        features = {}
        
        # Pitch (fundamental frequency) using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(self.audio_data, 
                                               fmin=librosa.note_to_hz('C2'), 
                                               fmax=librosa.note_to_hz('C7'),
                                               sr=self.sample_rate)
        f0 = f0[~np.isnan(f0)]  # Remove NaN values
        if len(f0) > 0:
            features['pitch_mean'] = np.mean(f0)
            features['pitch_std'] = np.std(f0)
            features['pitch_min'] = np.min(f0)
            features['pitch_max'] = np.max(f0)
            features['pitch_range'] = np.max(f0) - np.min(f0)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_range'] = 0
        
        # Energy/Intensity
        rmse = librosa.feature.rms(y=self.audio_data)[0]
        features['energy_mean'] = np.mean(rmse)
        features['energy_std'] = np.std(rmse)
        features['energy_max'] = np.max(rmse)
        
        # Speech rate (using zero-crossing rate as a proxy)
        zcr = librosa.feature.zero_crossing_rate(self.audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Speech rate estimation (words per minute) - simplified to avoid NLTK
        if self.transcription:
            # Simple word count without NLTK
            word_count = len(self.transcription.split())
            audio_duration = len(self.audio_data) / self.sample_rate
            features['speech_rate'] = word_count / (audio_duration / 60) if audio_duration > 0 else 0
        else:
            features['speech_rate'] = 0
        
        # Pauses analysis
        silence_threshold = 0.01
        is_silence = (rmse < silence_threshold)
        
        # Find silence segments
        silence_starts = np.where(np.logical_and(~is_silence[:-1], is_silence[1:]))[0]
        silence_ends = np.where(np.logical_and(is_silence[:-1], ~is_silence[1:]))[0]
        
        # Ensure we have matching starts and ends
        min_len = min(len(silence_starts), len(silence_ends))
        if min_len > 0:
            if silence_starts[0] > silence_ends[0]:
                silence_ends = silence_ends[:min_len]
            else:
                silence_starts = silence_starts[:min_len]
                silence_ends = silence_ends[:min_len]
            
            # Calculate pause durations in seconds
            pause_durations = [(silence_ends[i] - silence_starts[i]) / self.sample_rate for i in range(min_len)]
            features['pause_count'] = len(pause_durations)
            features['pause_mean_duration'] = np.mean(pause_durations) if pause_durations else 0
            features['pause_total_duration'] = np.sum(pause_durations) if pause_durations else 0
            features['pause_ratio'] = features['pause_total_duration'] / (len(self.audio_data) / self.sample_rate)
        else:
            features['pause_count'] = 0
            features['pause_mean_duration'] = 0
            features['pause_total_duration'] = 0
            features['pause_ratio'] = 0
        
        self.features = features
        print("Audio features extracted successfully")
        return features
    
    def analyze_text_sentiment(self):
        """Analyze the sentiment of the transcribed text"""
        if not self.transcription:
            print("No transcription available for sentiment analysis")
            return None

        sentiment_scores = {}

        # Basic sentiment analysis using TextBlob
        blob = TextBlob(self.transcription)
        sentiment_scores['polarity'] = blob.sentiment.polarity
        sentiment_scores['subjectivity'] = blob.sentiment.subjectivity

        # Determine current mood
        if sentiment_scores['polarity'] >= 0.5:
            sentiment_scores['mood'] = 'Happy'
        elif 0.1 <= sentiment_scores['polarity'] < 0.5:
            sentiment_scores['mood'] = 'Positive'
        elif -0.1 <= sentiment_scores['polarity'] < 0.1:
            sentiment_scores['mood'] = 'Neutral'
        elif -0.5 <= sentiment_scores['polarity'] < -0.1:
            sentiment_scores['mood'] = 'Negative'
        else:
            sentiment_scores['mood'] = 'Sad'

        tokens = self.transcription.lower().split()

        # Enhanced emotion word lists
        negative_words = ['sad', 'unhappy', 'depressed', 'miserable', 'disappointed', 'upset',
                        'anxious', 'worried', 'stressed', 'afraid', 'scared', 'nervous',
                        'terrible', 'horrible', 'awful', 'bad', 'worse', 'worst',
                        'problem', 'difficulty', 'struggle', 'pain', 'suffering',
                        'lonely', 'alone', 'isolated', 'abandoned', 'rejected',
                        'tired', 'exhausted', 'fatigue', 'hopeless', 'helpless']

        critical_indicators = {
            'self_harm': ['hurt myself', 'kill myself', 'suicide', 'end it all', 'self-harm', 
                        'cut myself', 'die', 'death', 'overdose', 'pills'],
            'time_indicators': ['morning', 'night', "can't sleep", 'insomnia', 'awake', 
                            '3am', '4am', 'dawn', 'dusk', 'midnight'],
            'emotional_sounds': ['cry', 'crying', 'sobbing', 'tears', 'weeping', 
                            'screaming', 'yelling', 'sigh', 'sighing'],
            'isolation': ['nobody', 'no one', 'alone', 'lonely', 'isolated', 
                        'by myself', 'abandoned', 'empty'],
            'urgency': ['help', 'emergency', 'crisis', 'urgent', 'immediate', 
                    'desperate', 'please', 'need']
        }

        # Count critical indicators
        sentiment_scores['critical_indicators'] = {}
        for category, words in critical_indicators.items():
            count = sum(1 for phrase in words if phrase in self.transcription.lower())
            sentiment_scores['critical_indicators'][category] = count

        # Risk level based on critical indicators
        total_critical = sum(sentiment_scores['critical_indicators'].values())
        if total_critical >= 3:
            sentiment_scores['risk_level'] = 'High'
        elif total_critical >= 1:
            sentiment_scores['risk_level'] = 'Medium'
        else:
            sentiment_scores['risk_level'] = 'Low'

        # Emotion words count
        positive_words = ['happy', 'joy', 'glad', 'pleased', 'delighted', 'satisfied',
                        'confident', 'motivated', 'inspired', 'proud', 'brave',
                        'calm', 'peaceful', 'relaxed', 'good', 'better', 'best',
                        'hope', 'optimistic', 'grateful', 'thankful', 'blessed',
                        'love', 'caring', 'kind', 'support', 'friend', 'family',
                        'enjoy', 'fun', 'laugh', 'smile', 'excited', 'interested']

        sentiment_scores['negative_word_count'] = sum(1 for word in tokens if word in negative_words)
        sentiment_scores['positive_word_count'] = sum(1 for word in tokens if word in positive_words)

        total_emotion_words = sentiment_scores['negative_word_count'] + sentiment_scores['positive_word_count']
        if total_emotion_words > 0:
            sentiment_scores['negative_ratio'] = sentiment_scores['negative_word_count'] / total_emotion_words
        else:
            sentiment_scores['negative_ratio'] = 0.5  # Neutral default

        # Sentence-level stats
        sentences = self.transcription.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        sentiment_scores['sentence_count'] = len(sentences)
        sentiment_scores['words_per_sentence'] = len(tokens) / len(sentences) if sentences else 0

        # First-person usage
        first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
        sentiment_scores['first_person_count'] = sum(1 for word in tokens if word in first_person_pronouns)
        sentiment_scores['first_person_ratio'] = sentiment_scores['first_person_count'] / len(tokens) if tokens else 0

        # --- 🔥 Emotion Classification with Pretrained Model ---
        if self.transcription:
            try:
                emotion_results = self.emotion_classifier(self.transcription)[0]
                emotion_distribution = {res['label']: res['score'] for res in emotion_results}

                emotion_map = {
                    'joy': 'Happy',
                    'neutral': 'Neutral',
                    'sadness': 'Sad',
                    'anger': 'Angry'
                }

                simplified_emotions = {'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Angry': 0}
                for label, score in emotion_distribution.items():
                    simplified_label = emotion_map.get(label.lower(), None)
                    if simplified_label:
                        simplified_emotions[simplified_label] += score

                total = sum(simplified_emotions.values())
                if total > 0:
                    for key in simplified_emotions:
                        simplified_emotions[key] = round((simplified_emotions[key] / total) * 100, 2)

                sentiment_scores['emotion_percentages'] = simplified_emotions
                print("\n🎭 Emotion Breakdown (%):")
                for emotion, pct in simplified_emotions.items():
                    print(f"{emotion}: {pct}%")
            except Exception as e:
                print(f"Error in emotion classification: {e}")
                sentiment_scores['emotion_percentages'] = {}

        self.sentiment_scores = sentiment_scores
        print(f"Text sentiment analysis complete - Current Mood: {sentiment_scores['mood']}")
        if sentiment_scores['risk_level'] in ['Medium', 'High']:
            print(f"⚠️ Risk Level: {sentiment_scores['risk_level']}")
        return sentiment_scores

    
    def calculate_mental_health_scores(self):
        """Calculate depression and anxiety risk scores based on features and sentiment"""
        if not self.features or not self.sentiment_scores:
            print("Features or sentiment scores missing. Run extract_audio_features() and analyze_text_sentiment() first.")
            return (0, 0)
        
        # This is a simplified scoring model based on research literature
        # A real-world implementation would use trained models on labeled data
        
        # Depression indicators and their weights
        depression_indicators = {
            'pitch_range': (-0.2, 150, 400),  # Lower pitch range often associated with depression
            'speech_rate': (-0.15, 100, 170),  # Slower speech may indicate depression
            'energy_mean': (-0.15, 0.05, 0.15),  # Lower energy/volume
            'pause_ratio': (0.1, 0.1, 0.4),  # More pauses
            'negative_ratio': (0.25, 0.3, 0.7),  # More negative sentiment
            'polarity': (-0.15, -0.5, 0.5),  # More negative polarity
            'first_person_ratio': (0.1, 0.05, 0.15)  # More self-focus
        }
        
        # Anxiety indicators and their weights
        anxiety_indicators = {
            'pitch_mean': (0.15, 120, 250),  # Higher pitch often associated with anxiety
            'pitch_std': (0.15, 10, 50),  # More pitch variation
            'speech_rate': (0.2, 150, 200),  # Faster speech may indicate anxiety
            'energy_std': (0.15, 0.02, 0.1),  # More variation in energy
            'zcr_mean': (0.1, 0.1, 0.3),  # Higher zero-crossing rate
            'pause_mean_duration': (-0.1, 0.2, 1.0),  # Shorter pauses
            'subjectivity': (0.15, 0.3, 0.7)  # More subjective language
        }
        
        # Helper function to normalize and score a feature
        def score_feature(value, weight, low_threshold, high_threshold):
            if low_threshold == high_threshold:
                return 0
            
            # Normalize to 0-1 range
            normalized = (value - low_threshold) / (high_threshold - low_threshold)
            normalized = max(0, min(1, normalized))  # Clamp to 0-1
            
            # If weight is negative, invert the normalized value
            if weight < 0:
                normalized = 1 - normalized
                weight = abs(weight)
            
            return normalized * weight
        
        # Calculate depression score
        depression_components = {}
        depression_score = 0
        for feature, (weight, low, high) in depression_indicators.items():
            if feature in self.features:
                value = self.features[feature]
            elif feature in self.sentiment_scores:
                value = self.sentiment_scores[feature]
            else:
                continue
                
            component_score = score_feature(value, weight, low, high)
            depression_components[feature] = (value, component_score)
            depression_score += component_score
        
        # Calculate anxiety score
        anxiety_components = {}
        anxiety_score = 0
        for feature, (weight, low, high) in anxiety_indicators.items():
            if feature in self.features:
                value = self.features[feature]
            elif feature in self.sentiment_scores:
                value = self.sentiment_scores[feature]
            else:
                continue
                
            component_score = score_feature(value, weight, low, high)
            anxiety_components[feature] = (value, component_score)
            anxiety_score += component_score
        
        # Normalize scores to 0-100 scale
        total_depression_weight = sum(abs(w) for w, _, _ in depression_indicators.values())
        total_anxiety_weight = sum(abs(w) for w, _, _ in anxiety_indicators.values())
        
        self.depression_score = min(100, max(0, (depression_score / total_depression_weight) * 100))
        self.anxiety_score = min(100, max(0, (anxiety_score / total_anxiety_weight) * 100))
        
        print(f"Depression Risk Score: {self.depression_score:.1f}/100")
        print(f"Anxiety Risk Score: {self.anxiety_score:.1f}/100")
        
        # Store the components for explanation
        self.depression_components = depression_components
        self.anxiety_components = anxiety_components
        
        return (self.depression_score, self.anxiety_score)
    
    def explain_scores(self):
        """Provide explanation for the depression and anxiety scores"""
        if not hasattr(self, 'depression_components') or not hasattr(self, 'anxiety_components'):
            print("No score components available. Run calculate_mental_health_scores() first.")
            return
        
        # print("\n--- Depression Score Explanation ---")
        # print(f"Overall Depression Risk Score: {self.depression_score:.1f}/100")
        # print("Contributing factors (sorted by impact):")
        
        # # Sort by absolute contribution
        # sorted_components = sorted(self.depression_components.items(), 
        #                           key=lambda x: abs(x[1][1]), 
        #                           reverse=True)
        
        # for feature, (value, contribution) in sorted_components:
        #     print(f"- {feature}: {value:.3f} (contribution: {contribution:.3f})")
        
        # print("\n--- Anxiety Score Explanation ---")
        # print(f"Overall Anxiety Risk Score: {self.anxiety_score:.1f}/100")
        # print("Contributing factors (sorted by impact):")
        
        # Sort by absolute contribution
        # sorted_components = sorted(self.anxiety_components.items(), 
        #                           key=lambda x: abs(x[1][1]), 
        #                           reverse=True)
        
        # for feature, (value, contribution) in sorted_components:
        #     print(f"- {feature}: {value:.3f} (contribution: {contribution:.3f})")
    
    def plot_results(self):
        """Plot visual representation of the scores and key features"""
        if not hasattr(self, 'depression_score') or not hasattr(self, 'anxiety_score'):
            print("No scores available. Run calculate_mental_health_scores() first.")
            return
        
        plt.figure(figsize=(15, 12))

        # 1. Plot: Audio Spectrogram
        plt.subplot(3, 1, 1)
        if self.audio_data is not None:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
            D = np.squeeze(D)  # Ensure 2D shape for the spectrogram
            librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Audio Spectrogram')

        # 2. Plot: Analysis Confidence (Depression and Anxiety Risk Scores)
        plt.subplot(3, 1, 2)
        scores = [self.depression_score, self.anxiety_score]
        labels = ['Depression Risk', 'Anxiety Risk']
        colors = ['#FF9999', '#9999FF']
        
        plt.bar(labels, scores, color=colors)
        plt.title('Mental Health Risk Assessment')
        plt.ylabel('Risk Score (0-100)')
        plt.ylim(0, 100)

        for i, score in enumerate(scores):
            plt.text(i, score + 2, f'{score:.1f}', ha='center')

        # 3. Plot: Mood Graph (Emotion Breakdown)
        plt.subplot(3, 1, 3)
        if 'emotion_percentages' in self.sentiment_scores:
            emotions = list(self.sentiment_scores['emotion_percentages'].keys())
            values = list(self.sentiment_scores['emotion_percentages'].values())
            colors = ['#FFD700', '#C0C0C0', '#87CEFA', '#FF6347']  # For Happy, Neutral, Sad, Angry

            plt.bar(emotions, values, color=colors)
            plt.title('Mood Breakdown')
            plt.ylabel('Percentage')
            plt.ylim(0, 100)

            for i, value in enumerate(values):
                plt.text(i, value + 2, f'{value:.1f}%', ha='center')

        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the voice mental health assessment"""
    assessor = VoiceMentalHealthAssessment()
    
    while True:
        print("\n===== Voice Mental Health Assessment System =====")
        print("1. Record audio (20 seconds)")
        print("2. Record audio (custom duration)")
        print("3. Load audio from file")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            assessor.record_audio(20)
            assessor.save_audio("last_recording.wav")
        
        elif choice == '2':
            try:
                duration = float(input("Enter recording duration in seconds: "))
                assessor.record_audio(duration)
                assessor.save_audio("last_recording.wav")
            except ValueError:
                print("Invalid duration. Please enter a number.")
        
        elif choice == '3':
            file_path = input('Enter the path to the audio file: ').strip('"').replace("\\", "/")
            assessor.load_audio_file(file_path)

        
        elif choice == '4':
            print("Exiting the program...")
            break
        
        else:
            print("Invalid choice. Please try again.")
            continue
        
        if assessor.audio_data is None:
            continue
        
        assessor.process_audio()
        assessor.explain_scores()
        assessor.plot_results()

if __name__ == "__main__":
    main()
