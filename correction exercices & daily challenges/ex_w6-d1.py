def display_course_info():
    course_info = """
    
    XP Exercises
    """

    exercises = {
        "Exercise 1": {
            "Title": "Traditional vs. Modern NLP: A Comparative Analysis",
            "Comparative Table": """
            Feature Engineering: Manual, handcrafted vs. Automatic, learned
            Word Representations: Static (e.g., TF-IDF) vs. Contextual (e.g., BERT)
            Model Architectures: Shallow (e.g., Naïve Bayes) vs. Deep (e.g., Transformers)
            Training Methodology: Task-specific training vs. Pre-training & fine-tuning
            Key Examples: Naïve Bayes, SVM, HMM vs. BERT, GPT, T5
            Advantages: Simpler, interpretable vs. High accuracy, transfer learning
            Disadvantages: Limited complexity, task specific, manual work vs. Computationally intensive, potential bias
            """,
            "Impact": "Modern NLP improves scalability and efficiency through pre-training and contextual embeddings."
        },
        "Exercise 2": {
            "Title": "LLM Architecture and Application Scenarios",
            "BERT": "Bidirectional Transformer encoder, used for search engine query understanding.",
            "GPT": "Unidirectional Transformer decoder, used for text generation.",
            "T5": "Encoder-decoder Transformer, used for machine translation."
        },
        "Exercise 3": {
            "Title": "The Benefits and Ethical Considerations of Pre-training",
            "Benefits": [
                "Improved Generalization",
                "Reduced Need for Labeled Data",
                "Faster Fine-tuning",
                "Transfer Learning",
                "Robustness"
            ],
            "Ethical Concerns": [
                "Bias",
                "Misinformation",
                "Misuse"
            ],
            "Mitigation Strategies": [
                "Data Curation",
                "Bias Detection and Mitigation",
                "Responsible AI Guidelines",
                "Transparency and Explainability"
            ]
        },
        "Exercise 4": {
            "Title": "Transformer Architecture Deep Dive",
            "Self-Attention": "Weighs importance of words in a sentence.",
            "Multi-Head Attention": "Focuses on different aspects of input simultaneously.",
            "Pre-training Objectives": [
                "MLM: Masks words and predicts them.",
                "CLM: Predicts the next word."
            ],
            "Transformer Model Selection": {
                "Sentiment Analysis": "Encoder-only (e.g., BERT)",
                "Chatbot": "Decoder-only (e.g., GPT)",
                "Machine Translation": "Encoder-decoder (e.g., T5)"
            },
            "Positional Encoding": "Incorporates word order information."
        },
        "Exercise 5": {
            "Title": "BERT Variations - Choose Your Detective",
            "Models": {
                "DistilBERT": "Optimized for speed and efficiency.",
                "RoBERTa": "Highest accuracy on complex text.",
                "XLM-RoBERTa": "Multilingual capabilities.",
                "ELECTRA": "Efficient pretraining and token replacement detection.",
                "ALBERT": "Efficient NLP in resource-constrained environments."
            }
        },
        "Exercise 6": {
            "Title": "Softmax Temperature - The Randomness Regulator",
            "Temperature Scenarios": {
                "0.2": "Deterministic, focused output.",
                "1.5": "Random, creative output.",
                "1": "Balanced randomness and certainty."
            },
            "Application Design": {
                "Bedtime Stories": "High temperature for creativity.",
                "Financial Reports": "Low temperature for accuracy."
            }
        }
    }

    print(course_info)
    for exercise, details in exercises.items():
        print(f"{exercise}: {details['Title']}")
        for key, value in details.items():
            if key != 'Title':
                print(f"  {key}: {value}")
        print("\n")

display_course_info()