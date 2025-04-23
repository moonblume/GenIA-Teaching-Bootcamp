# Day 2 Correction for GenAI & Machine Learning Bootcamp 2025

class Exercise:
    def __init__(self, title, goal, prompt_task, solution, justification=None, follow_up=None, memory_integration=None, mitigation_refinement=None):
        self.title = title
        self.goal = goal
        self.prompt_task = prompt_task
        self.solution = solution
        self.justification = justification
        self.follow_up = follow_up
        self.memory_integration = memory_integration
        self.mitigation_refinement = mitigation_refinement

    def display(self):
        print(f"Exercise Title: {self.title}\n")
        print(f"Goal: {self.goal}\n")
        print("Prompt Task:")
        print(self.prompt_task)
        print("\nSolution:")
        print(self.solution)
        if self.justification:
            print("\nJustification:")
            print(self.justification)
        if self.follow_up:
            print("\nFollow-Up Q&A:")
            print(self.follow_up)
        if self.memory_integration:
            print("\nMemory Integration:")
            print(self.memory_integration)
        if self.mitigation_refinement:
            print("\nMitigation & Refinement:")
            print(self.mitigation_refinement)
        print("\n" + "="*50 + "\n")

# Define exercises
exercises = [
    Exercise(
        title="Exercise 1: Debug a Faulty Chain-of-Thought",
        goal="Practice constructing and improving Chain-of-Thought prompts by spotting flaws in reasoning and refining step-by-step outputs.",
        prompt_task=(
            "A shop sells pencils at $0.75 each. If Alice buys 6 pencils and pays with a $5 bill, how much change does she get? "
            "Let’s solve this step-by-step.\n"
            "6 pencils × $0.75 = $4.75\n"
            "$5.00 - $4.75 = $0.50\n"
            "The change is $0.50."
        ),
        solution=(
            "Error: The multiplication is incorrect. 6 × $0.75 = $4.50, not $4.75.\n"
            "Correct Chain-of-Thought:\n"
            "Step 1: Calculate the total cost of 6 pencils: 6 × $0.75 = $4.50\n"
            "Step 2: Subtract from $5.00: $5.00 - $4.50 = $0.50\n"
            "Final Answer: $0.50"
        )
    ),
    Exercise(
        title="Exercise 2: Choose the Right Prompt Pattern",
        goal="Select the optimal prompting strategy for a real-world NLP use case and explain your choice.",
        prompt_task=(
            "Scenario: Categorize messages into one of:\n"
            "- Billing Issue\n"
            "- Technical Support\n"
            "- Account Access\n"
            "- Other"
        ),
        solution=(
            "Chosen Prompting Pattern: Few-Shot Prompting\n"
            "Prompt Example:\n"
            "You are a helpful assistant that categorizes customer support messages. Choose one category: Billing Issue, Technical Support, Account Access, or Other.\n\n"
            "Example 1:\n"
            "Message: 'I was charged twice this month.'\n"
            "Category: Billing Issue\n\n"
            "Example 2:\n"
            "Message: 'I can’t log into my account.'\n"
            "Category: Account Access\n\n"
            "Example 3:\n"
            "Message: 'My app keeps crashing when I open it.'\n"
            "Category: Technical Support\n\n"
            "Message: 'The verification email never arrived.'\n"
            "Category:"
        ),
        justification=(
            "Few-shot prompting provides clarity through examples, improving classification consistency and reducing ambiguity."
        )
    ),
    Exercise(
        title="Exercise 3: Use AlignedCoT to Compare Reasoning Paths",
        goal="Explore how Aligned Chain-of-Thought (AlignedCoT) reduces hallucination and improves answer reliability.",
        prompt_task=(
            "Problem:\n"
            "A gardener buys:\n"
            "- 2 small pots at $2 each\n"
            "- 3 medium pots at $4 each\n"
            "- 1 large pot at $6"
        ),
        solution=(
            "AlignedCoT Prompt:\n"
            "Path 1 (step-by-step):\n"
            "2 × $2 = $4\n"
            "3 × $4 = $12\n"
            "1 × $6 = $6\n"
            "Total = $4 + $12 + $6 = $22\n\n"
            "Path 2 (grouped calculation):\n"
            "Total = (2 × 2) + (3 × 4) + (1 × 6)\n"
            "Total = 4 + 12 + 6 = $22\n\n"
            "Comparison:\n"
            "Both methods result in $22.\n"
            "Final Answer: $22"
        )
    ),
    Exercise(
        title="Exercise 4: Design a Multi-Step Document Pipeline",
        goal="Apply prompt chaining and conditional logic to automate a real-world LLM workflow.",
        prompt_task=(
            "Scenario: Analyze academic research papers.\n\n"
            "Stage 1 - Identify Domain:\n"
            "Read the abstract below and determine the research domain (e.g., biology, physics, computer science, etc.).\n\n"
            "Abstract: [Insert abstract]\n"
            "Domain:\n\n"
            "Stage 2 - Extract Contributions:\n"
            "Extract the main contributions of this paper from the abstract.\n\n"
            "Abstract: [Insert abstract]\n"
            "Main Contributions:\n\n"
            "Stage 3 - Generate Follow-Up Question:\n"
            "Based on the identified contributions, suggest a relevant research question.\n\n"
            "Main Contributions: [Insert contributions]\n"
            "Research Question:\n\n"
            "Conditional Logic:\n"
            "If Domain == Biology, use biology-specific language in stage 3.\n"
            "Use context chaining to pass abstract → contributions → follow-up question."
        ),
        solution="N/A"  # No specific solution provided for this exercise
    ),
    Exercise(
        title="Exercise 5: Role Prompting to Reduce Bias",
        goal="Use role-based prompting to reduce assumptions and increase fairness in model responses.",
        prompt_task=(
            "Scenario:\n"
            "User wants career suggestions based on:\n"
            "Skills: empathetic, organized, good with people\n"
            "Interests: healthcare, helping others"
        ),
        solution=(
            "Basic Prompt (May Reinforce Biases):\n"
            "Based on the user’s skills and interests, suggest 3 possible careers.\n\n"
            "Role-Based Prompt (Fairness-Aware):\n"
            "You are an unbiased career counselor committed to inclusion and fairness.\n"
            "Suggest 3 career options based on the following:\n"
            "Skills: empathetic, organized, good with people\n"
            "Interests: healthcare, helping others\n\n"
            "Improvement:\n"
            "The role-based prompt minimizes the chance of gender/ethnicity-related stereotypes by explicitly stating fairness and inclusion."
        )
    ),
    Exercise(
        title="Exercise 6: Build a Conversational Agent with Context Memory",
        goal="Simulate memory in a chatbot using structured context chaining.",
        prompt_task=(
            "Scenario:\n"
            "Virtual health coach remembers previous conversations."
        ),
        solution=(
            "Technique: Structured history\n\n"
            "Stored Context:\n"
            "User: Alex\n"
            "Past Advice:\n"
            "- Sleep: Wind down 30 minutes before bed\n"
            "- Diet: Reduce caffeine after 2pm\n"
            "- Exercise: Light workouts 3x/week\n"
            "- Preferences: Plant-based meals, early workouts\n\n"
            "New Prompt:\n\n"
            "Alex is back with a follow-up question. Use the context below to respond.\n\n"
            "Context:\n"
            "Sleep: Wind down before bed\n"
            "Diet: Reduce caffeine after 2pm\n"
            "Exercise: Light workouts 3x/week\n"
            "Preferences: Plant-based meals, early workouts\n"
            "Message:\n"
            "\"I’ve been sticking to the plan but still feel tired in the mornings. What should I adjust?\"\n\n"
            "Response should:\n"
            "- Acknowledge effort\n"
            "- Suggest adjusting sleep duration, stress levels, screen exposure\n"
            "- Recommend sleep tracking or mindfulness routines"
        )
    )
]

# Display all exercises
for exercise in exercises:
    exercise.display()