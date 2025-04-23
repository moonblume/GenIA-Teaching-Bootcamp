# Day 1 Correction for GenAI & Machine Learning Bootcamp 2025

class Exercise:
    def __init__(self, title, scenario, issues, improved_prompt, reusability=None, context_table=None, features=None, justification=None, output_evaluation=None, revised_prompt=None, mitigation_strategies=None, domains=None):
        self.title = title
        self.scenario = scenario
        self.issues = issues
        self.improved_prompt = improved_prompt
        self.reusability = reusability
        self.context_table = context_table
        self.features = features
        self.justification = justification
        self.output_evaluation = output_evaluation
        self.revised_prompt = revised_prompt
        self.mitigation_strategies = mitigation_strategies
        self.domains = domains

    def display(self):
        print(f"Exercise Title: {self.title}\n")
        print(f"Scenario: {self.scenario}\n")
        if self.issues:
            print("Critical Issues:")
            for issue in self.issues:
                print(f"- {issue}")
            print()
        print("Improved Prompt:")
        print(self.improved_prompt)
        if self.reusability:
            print("\nReusability Add-On:")
            print(self.reusability)
        if self.context_table:
            print("\nContext Table:")
            for context in self.context_table:
                print(f"{context[0]}: {context[1]} - {context[2]}")
        if self.features:
            print("\nFeatures That Match the Style:")
            for feature in self.features:
                print(f"- {feature}")
        if self.justification:
            print("\nJustification:")
            print(self.justification)
        if self.output_evaluation:
            print("\nOutput Evaluation:")
            print(self.output_evaluation)
        if self.revised_prompt:
            print("\nRevised Prompt:")
            print(self.revised_prompt)
        if self.mitigation_strategies:
            print("\nMitigation Strategies:")
            for strategy in self.mitigation_strategies:
                print(f"- {strategy}")
        if self.domains:
            print("\nDomains Where Hallucinations Are Risky:")
            for domain in self.domains:
                print(f"- {domain}")
        print("\n" + "="*50 + "\n")

# Define exercises
exercises = [
    Exercise(
        title="Exercise 1: Rewrite and Optimize a Vague Prompt",
        scenario="You’re an AI Prompt Engineer at a productivity startup. Your manager asks you to use ChatGPT to create a LinkedIn post promoting their new focus app, FlowNest. They send you this vague prompt: 'Write something about productivity tips.'",
        issues=[
            "Lacks audience targeting – who is the message for?",
            "No format or content guidelines – how long? What kind of tips?",
            "No brand voice or tone specified – should it be formal, friendly, or fun?"
        ],
        improved_prompt=(
            "You are a content strategist at a productivity startup. Write a short LinkedIn post that shares 3 actionable productivity tips for busy tech professionals.\n"
            "- Use a friendly and professional tone\n"
            "- Present the tips as a bullet list\n"
            "- Keep the post under 280 characters total\n"
            "At the end, include the hashtag #FlowNestTips."
        ),
        reusability="Use this format for all future LinkedIn posts promoting productivity using FlowNest."
    ),
    Exercise(
        title="Exercise 2: Multi-Part Prompt for Quiz Generation",
        scenario="A 7th-grade science teacher uploads a short article on volcanic eruptions. You are tasked with writing a prompt for ChatGPT to create engaging educational content.",
        improved_prompt=(
            "You are a middle school science teacher assistant. Based on the article provided, perform the following tasks:\n"
            "- Summarize the article in 2 simple bullet points\n"
            "- Create 3 multiple-choice questions, each with 1 correct answer and 2 distractors\n"
            "- Use a friendly and age-appropriate tone (for 11–13-year-olds)\n"
            "- Output everything in a clear, well-organized format suitable for Google Slides"
        ),
        justification=(
            "Compared to the vague prompt 'Make a quiz for kids about this article':\n"
            "Audience is clearly defined (age 11–13)\n"
            "Specific tasks and structure are requested\n"
            "Output format is clearly identified (Google Slides-compatible)"
        )
    ),
    Exercise(
        title="Exercise 3: Add Context, Get Better Results",
        scenario="An intern is preparing a 3-minute summary for a monthly finance update. They give the prompt: 'Summarize this report.'",
        context_table=[
            ("Role", "Yes", "Act as a financial analyst"),
            ("Audience", "Yes", "For a non-technical executive team"),
            ("Purpose", "Yes", "To inform leadership in a 3-minute presentation"),
            ("Input Source", "Yes", "Specify: 'the attached monthly financial report'"),
            ("Format/Style", "Yes", "Bullet points with headlines"),
            ("Constraints", "Yes", "Max 3 key takeaways, each under 25 words")
        ],
        improved_prompt=(
            "You are a financial analyst preparing a short executive summary for a non-technical leadership team.\n"
            "Summarize the attached monthly financial report in 3 key takeaways, each under 25 words.\n"
            "Use bullet points with bolded headers. Keep the tone professional but easy to understand."
        )
    ),
    Exercise(
        title="Exercise 4: Match Prompt to Purpose",
        scenario="You’re building prompt templates for a customer support chatbot.",
        features=[
            "Tone is casual and empathetic",
            "Language mimics how a human would speak in a live chat"
        ],
        improved_prompt=(
            "You are a friendly and helpful customer support assistant. A user just reported that their order hasn’t arrived yet. Write a short, casual chat response that:\n"
            "- Acknowledges the issue\n"
            "- Apologizes\n"
            "- Assures the customer you’re investigating\n"
            "- Keeps the tone warm and understanding"
        ),
        justification=(
            "Conversational tone builds trust and eases frustration, which is essential for customer-facing interactions. "
            "Other styles (structured or functional) would sound too robotic."
        )
    ),
    Exercise(
        title="Exercise 5: Prompt Refinement Challenge – Control the Style, Structure, and Length",
        scenario="You need to write a product blurb for the PulseOne Mini smartwatch.",
        improved_prompt=(
            "Write a short product description for PulseOne Mini.\n"
            "Use exactly 3 bullet points\n"
            "Each point should mention 1 of the following: battery life, fitness tracking, Bluetooth\n"
            "Keep total word count under 50\n"
            "Tone should be friendly and engaging\n"
            "Do not include any extra features"
        ),
        output_evaluation=(
            "Example Output:\n"
            "- Tracks your workouts effortlessly\n"
            "- Connects via Bluetooth to all your devices\n"
            "- Battery lasts up to 3 days\n"
            "Word Count: 20\n"
            "✅ All constraints met"
        )
    ),
    Exercise(
        title="Exercise 6: Hallucination Spotting and Mitigation",
        scenario="The model falsely claims: 'Over 50% of marine species are projected to go extinct by 2050.'",
        improved_prompt=(
            "Based only on the content of the provided article, summarize the key points related to climate change and marine biodiversity.\n"
            "Do not include any information that is not explicitly stated in the source text."
        ),
        mitigation_strategies=[
            "Add explicit instructions to stay within source material",
            "Include a fallback mechanism like 'If the data is not in the text, state that it’s unavailable.'"
        ],
        domains=[
            "Healthcare: Incorrect summaries or made-up statistics can harm patients",
            "Legal: Misrepresentation of case law or policy language can lead to liability"
        ]
    )
]

# Display all exercises
for exercise in exercises:
    exercise.display()