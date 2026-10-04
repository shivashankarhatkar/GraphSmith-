from acme_supportos.models import get_model
from langchain_core.messages import SystemMessage, HumanMessage
SystemPrompt= [SystemMessage("You are Acme Cloud support.")]
history= [SystemPrompt]

model=get_model()
while True:
    user_input=input("User> ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    if user_input=="/reset":
        history=[SystemPrompt]
        continue

        

    history.append(HumanMessage(user_input))

    print("bot> ", end = "", flush=True)
    response_final = None
    
    for chunk in model.stream(history):
        print(chunk.text , end = "", flush=True)
        response_final=chunk if response_final is None else response_final+chunk 
    print()
    history.append(response_final)
    print(response_final.usage_metadata)


