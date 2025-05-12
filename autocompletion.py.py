import re
from collections import defaultdict
import heapq
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class PrecisionAutocomplete:
    def __init__(self):
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Language model storage
        self.ngram_model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        
        # Quality control parameters
        self.min_word_length = 3
        self.min_suggestion_score = 1.0
        self.common_words = ["will", "would", "should", "could", "might"]  # Quality fallbacks
        
        # Domain-specific knowledge (expand as needed)
        self.quality_phrases = {
            "python": ["code", "programming", "script", "developer"],
            "machine": ["learning", "vision", "translation"],
            "data": ["analysis", "science", "visualization"]
        }

    def preprocess_text(self, text):
        """Strict text cleaning with quality filters"""
        tokens = word_tokenize(text.lower())
        processed = []
        for token in tokens:
            # Validate word structure
            if (len(token) >= self.min_word_length and 
                token.isalpha() and 
                token not in self.stop_words):
                
                lemma = self.lemmatizer.lemmatize(token)
                processed.append(lemma)
                self.vocab.add(lemma)
        return processed

    def get_suggestions(self, prefix):
        """Generate high-quality suggestions only"""
        tokens = self.preprocess_text(prefix)
        if not tokens:
            return self.common_words[:5]
            
        last_word = tokens[-1]
        
        # Strategy 1: Check domain-specific phrases first
        if last_word in self.quality_phrases:
            return self.quality_phrases[last_word][:5]
        
        # Strategy 2: N-gram predictions with quality threshold
        candidates = defaultdict(float)
        for n in range(min(3, len(tokens)), 0, -1):
            context = tuple(tokens[-n:])
            for word, count in self.ngram_model.get(context, {}).items():
                score = count * (n * 0.5)  # Weight by context length
                if score >= self.min_suggestion_score:
                    candidates[word] = score
        
        # Strategy 3: Filter and sort valid suggestions
        valid_suggestions = [
            word for word in candidates.keys()
            if (word in self.vocab and 
                len(word) >= self.min_word_length and
                not word.startswith("'"))
        ]
        
        # Return best suggestions or fallback
        if valid_suggestions:
            return sorted(valid_suggestions, 
                        key=lambda x: -candidates[x])[:5]
        return self.common_words[:5]

    def update_model(self, text):
        """Learn only from quality input"""
        tokens = self.preprocess_text(text)
        for i in range(len(tokens) - 2):
            context = tuple(tokens[i:i+2])
            next_word = tokens[i+2]
            self.ngram_model[context][next_word] += 1

# Example usage
if __name__ == "__main__":
    ac = PrecisionAutocomplete()
    
    # Initial training with quality data
    training_data = [
        "Python programming is essential for data science",
        "Machine learning requires good python code",
        "Data visualization helps in analysis"
    ]
    for text in training_data:
        ac.update_model(text)
    
    print("Precision Autocomplete Ready")
    while True:
        user_input = input("\nCurrent text: ").strip()
        if user_input.lower() == 'exit':
            break
            
        suggestions = ac.get_suggestions(user_input)
        print("Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        ac.update_model(user_input)  # Learn from user input