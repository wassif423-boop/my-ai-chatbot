
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 My AI Chatbot")
st.write("Ask me anything!")

@st.cache_resource
def load_model():
    name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    return tokenizer, model

try:
    tokenizer, model = load_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Type your message...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Give simple answers."
            }
        ]

        messages.extend(st.session_state.messages[-10:])

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=120,
                        do_sample=True,
                        temperature=0.7
                    )

                answer = outputs[0][
                    inputs["input_ids"].shape[1]:
                ]

                reply = tokenizer.decode(
                    answer,
                    skip_special_tokens=True
                )

                st.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

except Exception as e:
    st.error("Model load nahi hua. Error details:")
    st.code(str(e))
