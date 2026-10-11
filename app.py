import os
import gradio as gr
from groq import Groq

# ---------------------------------
# Groq Model Configuration
# ---------------------------------
MODEL_NAME = "openai/gpt-oss-120b"


SYSTEM_PROMPT = (
    "You are a concise, helpful AI assistant focused on "
    "Data Science, Machine Learning, Generative AI, "
    "RAG, AI Agents and Agentic AI."
)

# ---------------------------------
# Chatbot Function
# ---------------------------------

def chat_fn(message, history):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "Error: GROQ_API_KEY is not configured. "
            "Please add it to your hosting platform's "
            "environment variables or secrets."
        )

    if not message or not message.strip():
        return ""

    try:
        client = Groq(api_key=api_key)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # Include previous conversation
        if history:
            for item in history:

                # Gradio messages format
                if isinstance(item, dict):
                    role = item.get("role")
                    content = item.get("content")

                    if (
                        role in ("user", "assistant")
                        and isinstance(content, str)
                        and content.strip()
                    ):
                        messages.append({
                            "role": role,
                            "content": content.strip()
                        })

                # Older Gradio tuple format
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    user_msg, assistant_msg = item

                    if user_msg:
                        messages.append({
                            "role": "user",
                            "content": str(user_msg)
                        })

                    if assistant_msg:
                        messages.append({
                            "role": "assistant",
                            "content": str(assistant_msg)
                        })

        # Current user message
        messages.append({
            "role": "user",
            "content": message.strip()
        })

        # Groq LLM response
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        error_msg = str(e)

        if "401" in error_msg or "Unauthorized" in error_msg:
            return "Error: Invalid GROQ_API_KEY."

        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return "Error: Groq API rate limit reached. Please try again later."

        else:
            return f"Groq API Error: {error_msg}"


# ---------------------------------
# Gradio Interface
# ---------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Data Science AI Assistant",
    description=(
        "Ask questions about Data Science, "
        "Machine Learning, Generative AI, "
        "RAG and AI Agents. Powered by Groq."
    ),
    examples=[
        "What is Machine Learning?",
        "Explain RAG with an example.",
        "What is Agentic AI?",
        "Explain transformers in simple terms.",
        "Difference between AI Agents and Agentic AI."
    ]
)


# ---------------------------------
# Launch Application
# ---------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7861))
    )
